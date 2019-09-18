// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "ngraph/op/constant.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

namespace nnfusion
{
    namespace graph
    {
        namespace pass
        {
            class RuntimeConstantFoldingPass : public GraphPassBase
            {
                int runtime_const_folding_iterate_once(
                    std::shared_ptr<Graph>& graph,
                    std::set<std::shared_ptr<GNode>>& blocklist_nodes)
                {
                    int folding_cnt = 0;
                    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
                    std::set<std::shared_ptr<GNode>> const_nodes = {};
                    std::set<std::shared_ptr<GNode>> down_streams = {};

                    // Find nodes with all constant upstream nodes
                    for (auto& it : nodes)
                        if (it->is_constant())
                        {
                            const_nodes.insert(it);
                            for (auto& edge : it->get_out_edges())
                            {
                                if (edge->is_control_edge())
                                    continue;

                                enforce(edge->get_src() == it);
                                auto dst = edge->get_dst();
                                if (blocklist_nodes.count(dst))
                                    continue;
                                if (down_streams.count(dst))
                                    continue;

                                bool inferable = true;
                                for (auto& in_edge : dst->get_in_edges())
                                {
                                    assert(in_edge->get_dst() == dst);
                                    auto p_const = std::dynamic_pointer_cast<ngraph::op::Constant>(
                                        in_edge->get_src()->get_op_ptr());
                                    if (!in_edge->get_src()->is_constant() ||
                                        p_const->is_parameter())
                                    {
                                        inferable = false;
                                        break;
                                    }
                                }
                                if (inferable)
                                    down_streams.insert(dst);
                            }
                        }

                    for (auto& it : down_streams)
                    {
                        auto eval_node = it->get_op_ptr();
                        LOG_INFO << ">> Found constant downstream node: " << it->get_name()
                                 << ", Op Type = " << eval_node->description();

                        bool const_infer_success = false;
                        std::vector<std::vector<char>> raw_inputs, raw_outputs;

                        // Prepare constant inputs from upstream_nodes
                        std::set<std::shared_ptr<GNode>> upstream_nodes;
                        for (auto& input : it->get_in_edges())
                        {
                            if (input->is_control_edge())
                                continue;
                            auto const_node = input->get_src();
                            LOG_INFO
                                << "  Input of constant downstream node: " << const_node->get_name()
                                << ", Op Type = " << const_node->get_op_ptr()->description() << "/"
                                << const_node->get_op_type();

                            enforce(input->get_dst() == it);
                            enforce(const_node->is_constant());
                            upstream_nodes.insert(const_node);

                            auto p_const = std::dynamic_pointer_cast<ngraph::op::Constant>(
                                const_node->get_op_ptr());
                            enforce(p_const != nullptr);
                            const void* ptr = p_const->get_data_ptr();
                            size_t length = p_const->get_data_size();
                            LOG_INFO << "  With Constant Input Node: " << p_const->get_name()
                                     << ", Memory Length = " << length;

                            std::vector<char> raw_input(length);
                            memcpy(raw_input.data(), ptr, length);
                            raw_inputs.emplace_back(std::move(raw_input));
                            enforce(raw_input.size() == 0);
                        }

                        // Prepare runtime backend
                        nnfusion::profiler::IProfilingRuntime::Pointer runtime = nullptr;
                        std::vector<shared_ptr<const KernelRegistration>> kernel_regs;

                        if (backend == "ROCm")
                        {
                            runtime = nnfusion::profiler::RocmDefaultRuntime::Runtime();
                            enforce(runtime->check_env());
                            kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                                eval_node->description(), ROCM_GPU, DT_FLOAT);
                            if (kernel_regs.size() == 0)
                                kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                                    eval_node->description(), CUDA_GPU, DT_FLOAT);
                        }
                        else if (backend == "CUDA")
                        {
                            runtime = nnfusion::profiler::CudaDefaultRuntime::Runtime();
                            enforce(runtime->check_env());
                            kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                                eval_node->description(), CUDA_GPU, DT_FLOAT);
                        }
                        else if (backend == "CPU")
                        {
                            runtime = nnfusion::profiler::ReferenceRuntime::Runtime();
                            enforce(runtime->check_env());
                            // TODO: need to fill correct kernel_regs list for CPU
                            enforce(false);
                        }
                        else
                        {
                            LOG_ERR << "Cannot Recognize Backend Type: " << backend;
                            enforce(false);
                        }

                        // Runtime node output inference
                        shared_ptr<KernelContext> ctx(new KernelContext(eval_node));
                        for (auto& kernel_reg : kernel_regs)
                        {
                            auto kernel = kernel_reg->m_factory(ctx);
                            if (!kernel->get_or_emit_source())
                                continue;

                            nnfusion::profiler::ProfilingContext::Pointer pctx =
                                make_shared<nnfusion::profiler::ProfilingContext>(kernel);

                            nnfusion::profiler::Profiler prof(runtime, pctx);
                            if (!prof.mixed_type_execute(raw_inputs, raw_outputs))
                                continue;

                            LOG_INFO << "  For node `" << eval_node->get_name()
                                     << "`: get runtime output results of size "
                                     << raw_outputs.size();
                            const_infer_success = true;
                            break;
                        }
                        if (!const_infer_success)
                        {
                            LOG_INFO << "  For node `" << eval_node->get_name()
                                     << "`: Cannot infer outputs, going to blacklist this node.";
                            blocklist_nodes.insert(it);
                            continue;
                        }

                        enforce(
                            raw_outputs.size() ==
                            1); // Only support single output; Multi-outputs lacks output-index properties in GNode.
#if 0                           // For Debug only
						printf("inputs = ");
						for (int i = 0; i < std::min(raw_inputs[0].size() / 4, 10LU); ++i)
							printf("%f ", ((float*)raw_inputs[0].data())[i]);
						puts("..");

						printf("outputs = ");
						for (int i = 0; i < std::min(raw_outputs[0].size() / 4, 10LU); ++i)
							printf("%f ", ((float*)raw_outputs[0].data())[i]);
						puts("..");
#endif
                        // Ensure output layout is as expected, replace eval_node with new_constant in place
                        enforce(raw_outputs.size() == eval_node->get_output_size());
                        for (int i = 0; i < eval_node->get_output_size(); ++i)
                        {
                            auto& shape = eval_node->get_output_shape(i);
                            auto& dtype = eval_node->get_output_element_type(i);
                            size_t memory = dtype.size();
                            for (auto& it : shape)
                                memory *= it;
                            enforce(memory == raw_outputs[i].size());

                            auto new_constant = std::make_shared<ngraph::op::Constant>(
                                dtype, shape, raw_outputs[i].data());
                            // new_constant->set_name("Constant_" + eval_node->get_name()); // not working?

                            // 1. remove upstream edges
                            auto upstream_edges = it->get_in_edges();
                            for (auto& edge : upstream_edges)
                            {
                                if (edge->is_control_edge())
                                    continue;
                                graph->remove_edge(edge);
                            }

                            // 2. remove upstream nodes with 0 out-degree
                            for (auto& node : upstream_nodes)
                                if (node->get_out_edges().size() == 0)
                                    graph->remove_node(node);

                            // 3. replace eval_node in place
                            it->reset_op_ptr(new_constant);

                            ++folding_cnt;
                            LOG_INFO << "  Finish folding " << folding_cnt
                                     << "th node: name = " << it->get_unique_name() << "/"
                                     << it->get_op_ptr()->get_name()
                                     << ", type = " << it->get_op_ptr()->description();
                            LOG_INFO << "";
                        }
                    }
                    return folding_cnt;
                }

            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override
                {
                    this->backend = getenv("NNFUSION_ENABLE_FOLDING_BACKEND")
                                        ? getenv("NNFUSION_ENABLE_FOLDING_BACKEND")
                                        : "";
                    if (this->backend == "")
                        return true;

                    static bool has_warning = false;
                    if (!has_warning)
                    {
                        has_warning = true;
                        LOG_INFO << "To disable Runtime Constant Folding: export "
                                    "NNFUSION_ENABLE_FOLDING_BACKEND=''";
                    }

                    LOG_INFO << "Runtime Constant Folding Pass starts up for Graph: "
                             << graph->get_name();

                    // Folding output nodes results in kernel_emitter crashes
                    std::set<std::shared_ptr<GNode>> blocklist_nodes = {};
                    for (auto& node : graph->get_outputs())
                        blocklist_nodes.insert(node);

                    int folding_cnt;
                    do
                    {
                        folding_cnt = runtime_const_folding_iterate_once(graph, blocklist_nodes);
                        LOG_INFO << ">> Runtime One Iteration Folds Infer-able Node Count: "
                                 << folding_cnt;
                    } while (folding_cnt > 0);
                    LOG_INFO << "";
                    LOG_INFO << ">> Runtime Constant Folding Pass ends for Graph: "
                             << graph->get_name();
                    LOG_INFO << "";
                    return true;
                }

            private:
                std::string backend;
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
