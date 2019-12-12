// Microsoft (c) 2019, Wenxiang Hu
#include "interpreter.hpp"
#include "nnfusion/engine/pass/cpu_codegenerator.hpp"
#include "nnfusion/engine/pass/create_fusion_block.hpp"
#include "nnfusion/engine/pass/cuda_codegenerator.hpp"
#include "nnfusion/engine/pass/device_dispatcher.hpp"
#include "nnfusion/engine/pass/elementwise_kernel_fusion.hpp"
#include "nnfusion/engine/pass/extract_graph_signature.hpp"
#include "nnfusion/engine/pass/kernel_selection.hpp"
#include "nnfusion/engine/pass/rocm_codegenerator.hpp"

#include <strings.h>
#include "pass/tensor/liveness_analysis.hpp"
#include "pass/tensor/tensor_memory_layout.hpp"
#include "pass/train/async_execution.hpp"
using namespace nnfusion::pass;

DECLARE_string(fdefault_device);

Interpreter::Interpreter()
    : m_trans_ctx(new InterpreterContext())
    , m_passes(new vector<shared_ptr<IInterpreterPass>>())
{
    // Todo: find another way
    auto dev_name = FLAGS_fdefault_device.c_str();
    DeviceType default_device;
    if (strcasecmp(dev_name, "ROCm") == 0)
        default_device = ROCM_GPU;
    else if (strcasecmp(dev_name, "CPU") == 0)
        default_device = GENERIC_CPU;
    else
        default_device = CUDA_GPU;

    // kernel selection
    m_passes->push_back(make_shared<DefaultDeviceDispatcher>());
    m_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
    m_passes->push_back(make_shared<DefaultKernelSelector>());

    /*
        This is disabled since we did use same stream for allreduce or applygradient;
        m_passes->push_back(make_shared<TrainningAsyncExecution>());
    */

    m_passes->push_back(make_shared<CreateFusionBlock>());
    m_passes->push_back(make_shared<ElementwiseKernelFusion>());
    m_passes->push_back(make_shared<TensorLivenessAnalysis>());
    // m_passes->push_back(make_shared<HostTensorAllocation>());
    m_passes->push_back(make_shared<AssignTensorMemoryLayout>(64, false));

    switch (default_device)
    {
    case CUDA_GPU: m_passes->push_back(make_shared<CudaCodeGenerator>()); break;

    case GENERIC_CPU: m_passes->push_back(make_shared<CpuCodeGenerator>()); break;

    case ROCM_GPU: m_passes->push_back(nnfusion::make_rocm_codegenerator()); break;

    default: m_passes->push_back(make_shared<CudaCodeGenerator>()); break;
    }
}

Interpreter::Interpreter(shared_ptr<vector<shared_ptr<IInterpreterPass>>> passes,
                         shared_ptr<InterpreterContext> ctx)
{
    this->m_passes = passes;
    this->m_trans_ctx = ctx;
}

bool Interpreter::translate(TranslationUnit::Pointer tu)
{
    CHECK_NOT_NULLPTR(m_passes);
    return IInterpreterPass::run_passes(*m_passes, m_trans_ctx, tu);
}

shared_ptr<TranslationUnitMap> Interpreter::translate(shared_ptr<graph::Graph> graph)
{
    // run graph passes
    nnfusion::pass::graph::GraphPass graph_passes;
    CHECK(graph_passes.run(graph));

    // TODO : multi graph ?
    m_trans_ctx->m_graphs.insert(graph);

    // Iterator through all nodes
    static interpreter::ExtractGraphSignature extract_global;
    shared_ptr<TranslationUnitMap> _tus(new TranslationUnitMap());

    // Deal with translation unit's program
    for (const auto& current_graph : m_trans_ctx->m_graphs)
    {
        shared_ptr<TranslationUnit> _tu(new TranslationUnit());
        _tus->emplace(current_graph, _tu);
        LOG(INFO) << "Translating graph:\t" << current_graph->get_name();

        _tu->program = nnfusion::ir::Program::create_single_basic_block_program();
        _tu->m_graph = current_graph;
        auto bb_main = _tu->program.get_entry();

        // extract output_names/constants/arg/out for _tu, m_variable_name_map for m_trans_ctx
        CHECK(extract_global.run(m_trans_ctx, _tu)) << "Error when extract global graph info.";

        // Translate the Node
        for (auto gnode : graph->get_ordered_ops())
        {
            // Generate Translated OP
            // <todo> not sure translated
            auto it = m_trans_ctx->m_node_inter_map.find(gnode);
            if (it == m_trans_ctx->m_node_inter_map.end())
            {
                nnfusion::ir::Instruction::Pointer ir(new nnfusion::ir::Instruction);
                ir->setGNode(gnode);
                // Attribute example code
                {
                    auto& attr = ir->Attr();
                    vector<TensorWrapper> in;
                    for (int i = 0; i < gnode->get_input_size(); i++)
                    {
                        shared_ptr<descriptor::Tensor> tv = gnode->get_input_tensor_ptr(i);
                        CHECK_NOT_NULLPTR(tv);
                        in.push_back(TensorWrapper(tv, tv->get_name()));
                    }
                    vector<TensorWrapper> out;
                    for (int i = 0; i < gnode->get_output_size(); i++)
                    {
                        shared_ptr<descriptor::Tensor> tv = gnode->get_output_tensor_ptr(i);
                        CHECK_NOT_NULLPTR(tv);
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

                ir->setName(gnode->get_name());
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
            LOG(INFO) << ss.str();
        }
         */
        translate(_tu);
    }
    return _tus;
}
