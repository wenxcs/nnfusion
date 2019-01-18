// Microsoft (c) 2019, Wenxiang Hu
#include "ngraph/runtime/nnfusion/nnfusion_functiontranslator.hpp"
#include "ngraph/runtime/nnfusion/intermediate/op_tbl.hpp"
#include "ngraph/runtime/nnfusion/pass/translator/ngraph_function_pass.hpp"
#include "ngraph/runtime/nnfusion/pass/translator/extract_function_signature.hpp"

using namespace ngraph::runtime::nnfusion;

ngraph::runtime::nnfusion::FunctionTranslator::FunctionTranslator()
    : m_trans_ctx(new FunctionTranslatorContext())
{
    m_passes.push_back(
        std::shared_ptr<IFunctionTranslatorPass>(new translator::ExtractFunctionSignature()));
}

std::shared_ptr<TranslationUnitMap> ngraph::runtime::nnfusion::FunctionTranslator::translate(
    std::shared_ptr<ngraph::Function> function)
{
    static translator::NgraphFunctionPass np;
    std::shared_ptr<TranslationUnitMap> _tus(new TranslationUnitMap());
    np.run(m_trans_ctx, nullptr, function);

    // Iterator through all functions
    for (const auto& p : m_trans_ctx->m_function_ordered_ops)
    {
        std::shared_ptr<TranslationUnit> _tu(new TranslationUnit());
        auto current_function = p.first;
        _tus->emplace(p.first, _tu);
        NGRAPH_DEBUG << "Translating function:\t" << current_function->get_name() << std::endl;

        if (!IFunctionTranslatorPass::run_passes(this->m_passes, m_trans_ctx, _tu, current_function))
            return false;

        // Translate the Node
        for (shared_ptr<Node> node : m_trans_ctx->m_function_ordered_ops.at(current_function))
        {
            vector<TensorWrapper> in;
            vector<string> node_input_names;
            vector<string> node_output_names;
            for (const descriptor::Input& input : node->get_inputs())
            {
                const descriptor::Output& output = input.get_output();
                shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                in.push_back(TensorWrapper(tv, m_trans_ctx->m_variable_name_map[tv->get_name()]));
                node_input_names.emplace_back(tv->get_name());
            }
            vector<TensorWrapper> out;
            for (const descriptor::Output& output : node->get_outputs())
            {
                shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                out.push_back(TensorWrapper(tv, m_trans_ctx->m_variable_name_map[tv->get_name()]));
                node_output_names.emplace_back(tv->get_name());
            }

            // Output debug info of node
            if (!node->is_parameter() && !node->is_constant())
            {
                NGRAPH_DEBUG << "Node:\t" << node->get_name() << "\t(";
                vector<string> parameter_nodes = node_input_names;
                parameter_nodes.insert(
                    parameter_nodes.end(), node_output_names.begin(), node_output_names.end());
                NGRAPH_DEBUG << join(parameter_nodes);
                NGRAPH_DEBUG << ")\n";
            }

            // Generate Translated OP
            // <todo> not sure translated
            auto it = m_trans_ctx->m_node_inter_map.find(node.get());
            if (it == m_trans_ctx->m_node_inter_map.end())
            {
                this->translate_node(node.get(), in, out);
            }
            _tu->inter_ops->push_back(m_trans_ctx->m_node_inter_map[node.get()]);
        }
    }
    return _tus;
}

bool ngraph::runtime::nnfusion::FunctionTranslator::translate_node(TRANS_ARGS)
{
    /*
    #define NGRAPH_OP(a, b) {type_index(typeid(b::a)), runtime::nnfusion::inter_op_##a##::translate}
        static const map<type_index, function<IntermediateOP(EMIT_ARGS)>> typeid_map{
    #include "ngraph/runtime/gpu/op/op_tbl.hpp"
        };
    #undef NGRAPH_OP
    */
    static const map<type_index, function<std::shared_ptr<IntermediateOP>(TRANS_ARGS)>> typeid_map{
        {type_index(typeid(ngraph::op::Reshape)),
         ngraph::runtime::nnfusion::intermediate::Reshape::translate},
        {type_index(typeid(ngraph::op::Parameter)),
         ngraph::runtime::nnfusion::intermediate::NoTrans::translate},
        {type_index(typeid(ngraph::op::Result)),
         ngraph::runtime::nnfusion::intermediate::Result::translate},
        {type_index(typeid(ngraph::op::Constant)),
         ngraph::runtime::nnfusion::intermediate::NoTrans::translate},
        {type_index(typeid(ngraph::op::Relu)),
         ngraph::runtime::nnfusion::intermediate::elementwise<ngraph::op::Relu>::translate},
        // {type_index(typeid(ngraph::op::Conv2D)), runtime::nnfusion::inter_op_conv2d::translate},
    };

    auto it = typeid_map.find(type_index(typeid(*node)));
    if (it == typeid_map.end())
    {
        // throw unsupported_op("Unsupported op '" + node->description() + "'");
        cout << "Unsupported op '" + node->description() + "'" << endl;
        return false;
    }
    cout << "Translate op '" + node->description() + "'" << endl;
    m_trans_ctx->m_node_inter_map.emplace(node, it->second(node, args, out));
    return true;
}