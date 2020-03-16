// Microsoft (c) 2019, NNFusion Team
#include "assign_async_info_pass.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::op;
using namespace nnfusion::pass::graph;
using namespace nnfusion::async;

DECLARE_bool(fadd_allreduce);
DECLARE_string(fdefault_device);
DECLARE_bool(frt_const_folding);
DEFINE_int32(fnum_stream, 1, "Number of streams. 0 means unlimited stream numbers.");

AssignAsyncInfoPass::AssignAsyncInfoPass()
{
    auto dev_name = FLAGS_fdefault_device.c_str();
    m_device = nnfusion::get_device_type(dev_name);
}

bool AssignAsyncInfoPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    auto async_manager = AsyncManagerFactory::get_async_manager(m_device);
    naive_assign_stream_info(async_manager, graph);
    assign_event_info(async_manager, graph);
    NNFUSION_LOG(INFO) << "run async-------------------------------";
    return true;
}

void AssignAsyncInfoPass::naive_assign_stream_info(AsyncManager* async_manager,
                                                   shared_ptr<Graph>& graph)
{
    bool allreduce_enable = FLAGS_fadd_allreduce;
    bool enable_rt_const_folding = FLAGS_frt_const_folding;
    int n_stream = FLAGS_fnum_stream;
    if (n_stream < 0)
        n_stream = 1;
    if (n_stream == 1)
    {
        for (auto gnode : graph->get_ordered_ops())
        {
            (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            async_info.execution_stream = async_manager->set_stream(0, "default");
        }
    }
    else
    {
        //const
        set<string> constant_vals;

        int count = 1;
        const GNodeVector& start = graph->get_outputs();
        // Stack of work to do.
        std::vector<std::shared_ptr<GNode>> stack(start.size());

        for (int i = 0; i < start.size(); ++i)
        {
            stack[i] = start[i];
        }

        std::vector<bool> visited(graph->get_max_node_id(), false);
        while (!stack.empty())
        {
            std::shared_ptr<GNode> gnode = stack.back();
            stack.pop_back();
            if (visited[gnode->get_id()])
            {
                continue;
            }
            visited[gnode->get_id()] = true;
            (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            // all constant ops use default stream
            if (gnode->get_op_type() == "Constant" || gnode->get_op_ptr()->is_parameter())
            {
                async_info.execution_stream = async_manager->set_stream(0, "default");
                if (enable_rt_const_folding)
                {
                    for (size_t i = 0; i < gnode->get_output_size(); ++i)
                    {
                        auto tensor = gnode->get_output_tensor_ptr(i);
                        constant_vals.insert(tensor->get_name());
                    }
                }
            }
            else
            {
                if (async_info.execution_stream == nullptr)
                {
                    async_info.execution_stream =
                        async_manager->set_stream(0, "base" + std::to_string(count));
                }
            }

            auto add_gnode = [&visited, &stack](std::shared_ptr<GNode> in_node) {
                if (!visited[in_node->get_id()])
                {
                    // Note; we must not mark as visited until we actually process it.
                    stack.push_back(in_node);
                }
            };

            size_t pre = stack.size();

            for (auto in_edge : gnode->get_in_edges())
            {
                add_gnode(in_edge->get_src());
            }

            if (stack.size() == pre)
            {
                if (n_stream == 0 || count < n_stream)
                    count += 1;
                else
                    count = 1;
            }
        }

        // If enable runtime constant folding, for cuda codegen, ops whose inputs are all constants are taken as constant ops.
        // And will be called in init() instead of kernel_entry(). So these ops use default stream as well.
        auto dt = async_manager->get_device_type();
        if (enable_rt_const_folding && (dt == CUDA_GPU || dt == ROCM_GPU))
        {
            for (auto gnode : graph->get_ordered_ops())
            {
                auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
                bool const_inputs = true;
                if (!gnode->get_op_ptr()->is_parameter() && !gnode->get_op_ptr()->is_output() &&
                    !gnode->is_constant())
                {
                    auto emitted_kernel =
                        (*gnode)["Kernel_Selection_Result"]
                            .as<pair<NNFusion_DeiveType, KernelEmitter::Pointer>>();

                    if (emitted_kernel.second->get_or_emit_source() == nullptr)
                    {
                        NNFUSION_LOG(NNFUSION_WARNING)
                            << "Kernel should be emitted before this pass:" << gnode->get_name();
                    }
                    auto kernel = emitted_kernel.second;
                    for (auto& in : kernel->m_context->input_names)
                    {
                        if (constant_vals.find(in) == constant_vals.end())
                        {
                            const_inputs = false;
                            break;
                        }
                    }
                    if (const_inputs)
                    {
                        for (auto& out : kernel->m_context->output_names)
                        {
                            constant_vals.insert(out);
                        }
                        async_info.execution_stream = async_manager->set_stream(0, "default");
                    }
                }
                else
                {
                    for (size_t i = 0; i < gnode->get_input_size(); ++i)
                    {
                        auto in = gnode->get_input_tensor_ptr(i)->get_name();
                        if (constant_vals.find(in) == constant_vals.end())
                        {
                            const_inputs = false;
                            break;
                        }
                    }
                    if (const_inputs)
                    {
                        for (size_t i = 0; i < gnode->get_output_size(); ++i)
                        {
                            auto out = gnode->get_output_tensor_ptr(i)->get_name();
                            constant_vals.insert(out);
                        }
                        async_info.execution_stream = async_manager->set_stream(0, "default");
                    }
                }
            }
        }
    }
    NNFUSION_LOG(INFO) << "assign stream info-------------------------------";
}

void AssignAsyncInfoPass::assign_event_info(AsyncManager* async_manager,
                                            std::shared_ptr<Graph>& graph)
{
    for (auto gnode : graph->get_ordered_ops())
    {
        NNFUSION_CHECK((*gnode)["Async_info"].is_valid());
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        auto stream = async_info.execution_stream;
        NNFUSION_CHECK_NOT_NULLPTR(stream);
        for (auto& edge : gnode->get_in_edges())
        {
            auto input_gnode = edge->get_src();
            // constant ops are in xxx_init() of generated code,
            // so there is no need to add event.
            if (input_gnode->get_op_ptr()->is_constant() ||
                input_gnode->get_op_ptr()->is_parameter())
            {
                continue;
            }
            auto& input_async_info = (*input_gnode)["Async_info"].as<AsyncExecutionInfo>();
            auto input_stream = input_async_info.execution_stream;
            if (input_stream->get_device_name() == stream->get_device_name())
            {
                if (input_stream->get_stream_id() != stream->get_stream_id())
                {
                    // Cuda streams perform implicite sychronization with default(0) stream,
                    // so there is no need to add event emplicitely.
                    if ((stream->get_device_type() == CUDA_GPU ||
                         stream->get_device_type() == ROCM_GPU) &&
                        (stream->is_default_stream() || input_stream->is_default_stream()))
                    {
                        continue;
                    }
                    if (input_async_info.record_event == nullptr)
                    {
                        input_async_info.record_event =
                            async_manager->set_event(input_stream, input_gnode->get_op_ptr());
                    }
                    async_info.wait_events.push_back(input_async_info.record_event);
                }
            }
            // todo: support cross-device events
            else
            {
                throw nnfusion::errors::NotSupported("Cross-device event is not supported.");
            }
        }
    }
    NNFUSION_LOG(INFO) << "assign event info-------------------------------";
}