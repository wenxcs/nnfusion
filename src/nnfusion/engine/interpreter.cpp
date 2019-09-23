// Microsoft (c) 2019, Wenxiang Hu
#include "interpreter.hpp"
#include "nnfusion/engine/pass/cpu_codegenerator.hpp"
#include "nnfusion/engine/pass/create_fusion_block.hpp"
#include "nnfusion/engine/pass/cuda_codegenerator.hpp"
#include "nnfusion/engine/pass/device_dispatcher.hpp"
#include "nnfusion/engine/pass/extract_function_signature.hpp"
#include "nnfusion/engine/pass/extract_graph_signature.hpp"
#include "nnfusion/engine/pass/kernel_selection.hpp"
#include "nnfusion/engine/pass/ngraph_function_pass.hpp"
#include "nnfusion/engine/pass/rocm_codegenerator.hpp"

#include "pass/tensor/host_tensor_allocation.hpp"
#include "pass/tensor/tensor_memory_layout.hpp"
#include "pass/train/async_execution.hpp"

using namespace nnfusion::pass;

Interpreter::Interpreter()
    : m_trans_ctx(new InterpreterContext())
    , m_passes(new vector<shared_ptr<IInterpreterPass>>())
{
    // kernel selection
    m_passes->push_back(make_shared<DefaultDeviceDispatcher>());
    m_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
    m_passes->push_back(make_shared<DefaultKernelSelector>());

    /*
        This is disabled since we did use same stream for allreduce or applygradient;
        m_passes->push_back(make_shared<TrainningAsyncExecution>());
    */

    // CUDA
    m_passes->push_back(make_shared<HostTensorAllocation>(CUDA_GPU));
    m_passes->push_back(make_shared<AssignTensorMemoryLayout>(64, true, CUDA_GPU));
    m_passes->push_back(make_shared<CreateFusionBlock>());
    m_passes->push_back(make_shared<CudaCodeGenerator>());

    /*
    // CPU
    m_passes->push_back(make_shared<AssignTensorMemoryLayout>(64, false, GENERIC_CPU));
    m_passes->push_back(make_shared<CpuCodeGenerator>());
    // ROCM
    m_passes->push_back(make_shared<AssignTensorMemoryLayout>(64, false, ROCM_GPU));
    m_passes->push_back(make_shared<DefaultDeviceDispatcher>(DefaultDeviceDispatcher()));
    m_passes->push_back(make_shared<CreateFusionBlock>(CreateFusionBlock()));
    m_passes->push_back(make_shared<ProfilingBasedKernelSelector>(ProfilingBasedKernelSelector()));
    m_passes->push_back(make_shared<CpuCodeGenerator>(CpuCodeGenerator()));
    m_passes->push_back(make_shared<CudaCodeGenerator>(CudaCodeGenerator()));
    m_passes->push_back(nnfusion::make_rocm_codegenerator());
    */
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

        _tu->program = nnfusion::ir::Program::create_single_basic_block_program();
        _tu->m_function = current_function;
        auto bb_main = _tu->program.get_entry();

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

                    // attr.ts_("INPUT", std::move(in))->ts_("OUTPUT", std::move(out));
                }

                // Tag example
                {
                    auto& INS = *ir;
                    INS["DEBUG"] = 1;
                    auto res = INS["DEBUG"].as<int>();
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

shared_ptr<GraphTranslationUnitMap> Interpreter::translate(shared_ptr<graph::Graph> graph)
{
    // run graph passes
    nnfusion::graph::pass::GraphPass graph_passes;
    enforce(graph_passes.run(graph));
    shared_ptr<TranslationUnit> graph_tu(new TranslationUnit());
    graph_tu->m_graph = graph;
    // todo: multi graph???
    m_trans_ctx->m_graphs.insert(graph);

    // Iterator through all nodes
    static interpreter::ExtractGraphSignature extract_global;
    shared_ptr<GraphTranslationUnitMap> _tus(new GraphTranslationUnitMap());

    // Deal with translation unit's program
    for (const auto& current_graph : m_trans_ctx->m_graphs)
    {
        shared_ptr<TranslationUnit> _tu(new TranslationUnit());
        _tus->emplace(current_graph, _tu);
        LOG_INFO << "Translating graph:\t" << current_graph->get_name() << endl;

        _tu->program = nnfusion::ir::Program::create_single_basic_block_program();
        _tu->m_graph = current_graph;
        auto bb_main = _tu->program.get_entry();

        enforce(extract_global.run(m_trans_ctx, _tu)) << "Error when extract global graph info.";

        // Translate the Node
        for (auto gnode : graph->get_ordered_ops())
        {
            // Generate Translated OP
            // <todo> not sure translated
            auto node = gnode->get_op_ptr();
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

                    //attr.ts_("INPUT", std::move(in))->ts_("OUTPUT", std::move(out));
                }

                // Tag example
                {
                    auto& INS = *ir;
                    INS["DEBUG"] = 1;
                    auto res = INS["DEBUG"].as<int>();
                }

                // move fusion group tags to intructions
                if ((*gnode)["elem_group_id"].is_valid() || (*gnode)["fusion_group_id"].is_valid())
                {
                    ir->copy_tags_from(*gnode);
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