// Microsoft (c) 2019, Wenxiang Hu
#include "interpreter.hpp"
#include "nnfusion/engine/pass/cuda_codegenerator.hpp"
#include "nnfusion/engine/pass/device_dispatcher.hpp"
#include "nnfusion/engine/pass/extract_function_signature.hpp"
#include "nnfusion/engine/pass/ngraph_function_pass.hpp"

Interpreter::Interpreter()
    : m_trans_ctx(new InterpreterContext())
    , m_passes(new vector<shared_ptr<IInterpreterPass>>())
{
    m_passes->push_back(make_shared<DefaultDeviceDispatcher>(DefaultDeviceDispatcher()));
    m_passes->push_back(make_shared<CudaCodeGenerator>(CudaCodeGenerator()));
}

Interpreter::Interpreter(shared_ptr<vector<shared_ptr<IInterpreterPass>>> passes,
                         shared_ptr<InterpreterContext> ctx)
{
    this->m_passes = passes;
    this->m_trans_ctx = ctx;
}

bool Interpreter::translate(TranslationUnit::Pointer tu)
{
    enforce_not_nullptr(m_passes);
    return IInterpreterPass::run_passes(*m_passes, m_trans_ctx, tu);
}

shared_ptr<TranslationUnitMap> Interpreter::translate(shared_ptr<ngraph::Function> function)
{
    /*  Run original Ngraph Passes */
    static interpreter::NgraphFunctionPass ngraph_passes;
    static interpreter::ExtractFunctionSignature extract_global;
    shared_ptr<TranslationUnitMap> _tus(new TranslationUnitMap());
    shared_ptr<TranslationUnit> ngraph_tu(new TranslationUnit());
    ngraph_tu->m_function = function;
    enforce(ngraph_passes.run(m_trans_ctx, ngraph_tu));
    // Iterator through all functions

    // Deal with translation unit's program
    for (const auto& p : m_trans_ctx->m_function_ordered_ops)
    {
        shared_ptr<TranslationUnit> _tu(new TranslationUnit());
        auto current_function = p.first;
        _tus->emplace(p.first, _tu);
        LOG_INFO << "Translating function:\t" << current_function->get_name() << endl;

        BasicBlock::pointer bb_main(new BasicBlock);
        bb_main->prior = nullptr;
        bb_main->next = nullptr;
        _tu->program.entry = bb_main;
        _tu->program.exit = bb_main;
        _tu->m_function = current_function;

        enforce(extract_global.run(m_trans_ctx, _tu)) << "Error when extract global graph info.";

        // Translate the Node
        for (shared_ptr<Node> node : m_trans_ctx->m_function_ordered_ops.at(current_function))
        {
            // Generate Translated OP
            // <todo> not sure translated
            auto it = m_trans_ctx->m_node_inter_map.find(node);
            if (it == m_trans_ctx->m_node_inter_map.end())
            {
                nnfusion::ir::Instruction::Pointer ir(new nnfusion::ir::Instruction);
                ir->setOperatorDef(node);
                // Attribute example
                {
                    auto& attr = ir->Attr();
                    vector<TensorWrapper> in;
                    for (const descriptor::Input& input : node->get_inputs())
                    {
                        const descriptor::Output& output = input.get_output();
                        shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                        enforce_not_nullptr(tv);
                        in.push_back(TensorWrapper(tv, tv->get_name()));
                    }
                    vector<TensorWrapper> out;
                    for (const descriptor::Output& output : node->get_outputs())
                    {
                        shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                        enforce_not_nullptr(tv);
                        out.push_back(TensorWrapper(tv, tv->get_name()));
                    }

                    attr.ts_("INPUT", std::move(in))->ts_("OUTPUT", std::move(out));
                }

                // Tag example
                {
                    auto& tag = ir->Tag();
                    tag.Set<int>("DEBUG", 1);
                }
                ir->setName(node->get_name());
                bb_main->push_back(ir);
            }
        }

        /*
        for (auto& ins : *bb_main)
        {
            std::stringstream ss;
            ss << ins->name() << "\t { ";
            ss << "INPUT:{";
            for(auto& in: ins->Attr().ts("INPUT"))
            {
                ss << in.get_name() << ", ";
            }
            ss << "}, ";
            ss << "OUTPUT:{";
            for(auto& in: ins->Attr().ts("OUTPUT"))
            {
                ss << in.get_name() << ", ";
            }
            ss << "}, (tag:)";
            ss << " DEBUG : " << ins->Tag().Get<int>("DEBUG") << " }";
            LOG_INFO << ss.str();
        }
        */

        translate(_tu);
    }
    return _tus;
}

ir::Operator_p Interpreter::translate_node(shared_ptr<Node> node)
{
    const map<type_index, function<ir::Operator_p(shared_ptr<Node>)>> typeid_map{};
    auto it = typeid_map.find(type_index(typeid(*node)));
    if (it == typeid_map.end())
    {
        LOG_WARN << "Unsupported op '" + node->description() + "', using Anyop instead." << endl;
        return nullptr; //ir::Anyop::translate(node);
    }
    LOG_INFO << "Translate op '" + node->description() + "'" << endl;
    return it->second(node);
}