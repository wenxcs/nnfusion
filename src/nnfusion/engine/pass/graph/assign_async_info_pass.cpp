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
    if (m_device == CUDA_GPU || m_device == ROCM_GPU)
    {
        auto CPU_async_manager = AsyncManagerFactory::get_async_manager(GENERIC_CPU);
        auto CUDA_async_manager = AsyncManagerFactory::get_async_manager(CUDA_GPU);
        assign_thread_info(CPU_async_manager, graph);
        naive_assign_stream_info(CUDA_async_manager, graph);
        assign_event_info(CUDA_async_manager, CPU_async_manager, graph);
    }
    else if (m_device == GENERIC_CPU)
    {
        auto CPU_async_manager = AsyncManagerFactory::get_async_manager(GENERIC_CPU);
        naive_assign_stream_info(CPU_async_manager, graph);
        assign_event_info(nullptr, CPU_async_manager, graph);
    }
    else
    {
        throw nnfusion::errors::NotSupported("Not Supported Device Type");
    }

    NNFUSION_LOG(INFO) << "run async-------------------------------";
    return true;
}

// assign thread for cuda
void AssignAsyncInfoPass::assign_thread_info(nnfusion::async::AsyncManager* CPU_async_manager,
                                             shared_ptr<Graph>& graph)
{
    bool enable_rt_const_folding = FLAGS_frt_const_folding;

    size_t num_async_node = 0;
    static const std::unordered_set<std::string> async_node = {"AllReduce"};

    for (auto gnode : graph->get_ordered_ops())
    {
        if (async_node.find(gnode->get_op_type()) != async_node.end())
        {
            (*gnode)["is_async_node"] = true;
            num_async_node += 1;
        }
    }

    if (num_async_node == 0)
    {
        for (auto gnode : graph->get_ordered_ops())
        {
            (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            async_info.execution_thread = CPU_async_manager->set_stream(0, "default");
        }
    }
    else
    {
        //const
        set<string> constant_vals;

        int count = 0;
        int max_num_async_thread = 1000; // could be modified.

        for (auto gnode : graph->get_ordered_ops())
        {
            if (!(*gnode)["Async_info"].is_valid())
                (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            if (async_info.execution_thread == nullptr)
            {
                // constant and parameter ops are in cuda_init(), and use default/main thread
                if (gnode->get_op_type() == "Constant" || gnode->get_op_ptr()->is_parameter())
                {
                    async_info.execution_thread = CPU_async_manager->set_stream(0, "default");

                    if (enable_rt_const_folding)
                    {
                        for (size_t i = 0; i < gnode->get_output_size(); ++i)
                        {
                            auto tensor = gnode->get_output_tensor_ptr(i);
                            constant_vals.insert(tensor->get_name());
                        }
                    }
                }
                else if ((*gnode)["is_async_node"].is_valid() &&
                         (*gnode)["is_async_node"].as<bool>())
                {
                    async_info.execution_thread =
                        CPU_async_manager->set_stream(0, "async_" + to_string(count));
                    if (count < max_num_async_thread)
                        count += 1;
                    std::vector<std::shared_ptr<nnfusion::graph::GNode>> stack;
                    std::unordered_set<std::shared_ptr<nnfusion::graph::GNode>> same_thread{gnode};

                    auto add_gnode = [&stack, &same_thread](std::shared_ptr<GNode> out_gnode) {
                        if (!(*out_gnode)["Async_info"].is_valid())
                            (*out_gnode)["Async_info"] = AsyncExecutionInfo();
                        auto& out_async_info = (*out_gnode)["Async_info"].as<AsyncExecutionInfo>();
                        bool same = true;
                        for (auto& in_edge : out_gnode->get_in_edges())
                        {
                            auto in_node = in_edge->get_src();
                            if (same_thread.find(in_node) == same_thread.end())
                            {
                                same = false;
                                break;
                            }
                        }
                        if (same && out_async_info.execution_thread == nullptr &&
                            !(*out_gnode)["is_async_node"].is_valid())
                        {
                            stack.push_back(out_gnode);
                            same_thread.insert(out_gnode);
                        }
                    };

                    for (auto& out_edge : gnode->get_out_edges())
                    {
                        add_gnode(out_edge->get_dst());
                    }

                    while (!stack.empty())
                    {
                        std::shared_ptr<GNode> cur_gnode = stack.back();
                        stack.pop_back();
                        auto& cur_async_info = (*cur_gnode)["Async_info"].as<AsyncExecutionInfo>();
                        cur_async_info.execution_thread = async_info.execution_thread;
                        for (auto& out_edge : cur_gnode->get_out_edges())
                        {
                            add_gnode(out_edge->get_dst());
                        }
                    }
                }
                else
                {
                    for (auto& in_edge : gnode->get_in_edges())
                    {
                        auto in_gnode = in_edge->get_src();
                        if (in_gnode->get_op_type() != "Constant" &&
                            !in_gnode->get_op_ptr()->is_parameter())
                        {
                            auto& in_async_info =
                                (*in_gnode)["Async_info"].as<AsyncExecutionInfo>();
                            auto in_thread = in_async_info.execution_thread;
                            if (in_thread != nullptr)
                            {
                                async_info.execution_thread = in_thread;
                                break;
                            }
                        }
                    }

                    if (async_info.execution_thread == nullptr)
                    {
                        async_info.execution_thread = CPU_async_manager->set_stream(0, "base");
                    }
                }
            }
        }

        // If enable runtime constant folding, for cuda codegen, ops whose inputs are all constants are taken as constant ops.
        // And will be called in init() instead of kernel_entry(). So these ops use default thread as well.

        if (enable_rt_const_folding)
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
                            .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();

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
                        async_info.execution_thread = CPU_async_manager->set_stream(0, "default");
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
                        async_info.execution_thread = CPU_async_manager->set_stream(0, "default");
                    }
                }
            }
        }
    }

    NNFUSION_LOG(INFO) << "assign thread info-------------------------------";
}

// assign stream of gpu or thread of cpu
void AssignAsyncInfoPass::naive_assign_stream_info(AsyncManager* async_manager,
                                                   shared_ptr<Graph>& graph)
{
    bool allreduce_enable = FLAGS_fadd_allreduce;
    bool enable_rt_const_folding = FLAGS_frt_const_folding;
    int n_stream = FLAGS_fnum_stream;
    if (n_stream < 0)
        n_stream = 1;
    // if (n_stream == 1)
    // {
    //     for (auto gnode : graph->get_ordered_ops())
    //     {
    //         if (!(*gnode)["Async_info"].is_valid())
    //             (*gnode)["Async_info"] = AsyncExecutionInfo();
    //         auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
    //         async_info.execution_stream = async_manager->set_stream(0, "default");
    //     }
    // }
    // else
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
            if (!(*gnode)["Async_info"].is_valid())
                (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            // all constant ops use default stream
            if (gnode->get_op_type() == "Constant" || gnode->get_op_ptr()->is_parameter())
            {
                if (m_device == GENERIC_CPU)
                    async_info.execution_thread = async_manager->set_stream(0, "default");
                else if (m_device == CUDA_GPU || m_device == ROCM_GPU)
                {
                    auto thread = async_info.execution_thread;
                    NNFUSION_CHECK(thread != nullptr);
                    NNFUSION_CHECK(thread->is_default_stream());
                    async_info.execution_stream = async_manager->set_stream(0, "default");
                }

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
                    if (m_device == GENERIC_CPU)
                    {
                        async_info.execution_thread =
                            async_manager->set_stream(0, "base" + std::to_string(count));
                    }
                    else if (m_device == CUDA_GPU || m_device == ROCM_GPU)
                    {
                        auto thread = async_info.execution_thread;
                        NNFUSION_CHECK(thread != nullptr);
                        std::string thread_symbol = thread->get_symbol() + "_thread_";
                        async_info.execution_stream = async_manager->set_stream(
                            0, thread_symbol + "base" + std::to_string(count));
                    }
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
                            .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();

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
                        auto thread = async_info.execution_thread;
                        NNFUSION_CHECK(thread != nullptr);
                        NNFUSION_CHECK(thread->is_default_stream());
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
                        auto thread = async_info.execution_thread;
                        NNFUSION_CHECK(thread != nullptr);
                        NNFUSION_CHECK(thread->is_default_stream());
                        async_info.execution_stream = async_manager->set_stream(0, "default");
                    }
                }
            }
        }
    }
    NNFUSION_LOG(INFO) << "assign thread or stream info-------------------------------";
}

// assign event of gpu or barrier of cpu
void AssignAsyncInfoPass::assign_event_info(nnfusion::async::AsyncManager* CUDA_async_manager,
                                            nnfusion::async::AsyncManager* CPU_async_manager,
                                            std::shared_ptr<Graph>& graph)
{
    for (auto gnode : graph->get_ordered_ops())
    {
        NNFUSION_CHECK((*gnode)["Async_info"].is_valid());
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        auto stream = async_info.execution_stream;
        auto thread = async_info.execution_thread;

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
            auto input_thread = input_async_info.execution_thread;

            if (input_thread->get_device_name() == thread->get_device_name())
            {
                if (input_thread->get_stream_id() != thread->get_stream_id())
                {
                    if (input_async_info.notify_barrier == nullptr)
                    {
                        input_async_info.notify_barrier =
                            CPU_async_manager->set_event(input_thread, input_gnode->get_op_ptr());
                    }
                    async_info.wait_barriers.push_back(input_async_info.notify_barrier);
                }
            }
            // todo: support cross-device event
            else
            {
                throw nnfusion::errors::NotSupported("Cross-device barrier is not supported.");
            }

            if (m_device == CUDA_GPU || m_device == ROCM_GPU)
            {
                if (input_stream->get_device_name() == stream->get_device_name())
                {
                    if (input_stream->get_stream_id() != stream->get_stream_id())
                    {
                        // Cuda streams perform implicite sychronization with default(0) stream,
                        // so there is no need to add event emplicitely.
                        if (stream->is_default_stream() || input_stream->is_default_stream())
                        {
                            continue;
                        }
                        if (input_async_info.record_event == nullptr)
                        {
                            input_async_info.record_event = CUDA_async_manager->set_event(
                                input_stream, input_gnode->get_op_ptr());
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
    }
    NNFUSION_LOG(INFO) << "assign event info-------------------------------";
}