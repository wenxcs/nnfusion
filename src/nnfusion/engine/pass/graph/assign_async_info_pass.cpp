// Microsoft (c) 2019, NNFusion Team
#include "assign_async_info_pass.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::op;
using namespace nnfusion::pass::graph;
using namespace nnfusion::async;
using namespace nnfusion::kernels::cuda;

DECLARE_bool(fadd_allreduce);
DECLARE_string(fdefault_device);
DECLARE_bool(frt_const_folding);
DEFINE_int32(fnum_stream, 1, "Number of streams. 0 means unlimited stream numbers.");
DECLARE_int32(fnum_device);
DEFINE_bool(fuse_default_stream, true, "Use default stream.");

AssignAsyncInfoPass::AssignAsyncInfoPass()
{
}

bool AssignAsyncInfoPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    auto dev_name = FLAGS_fdefault_device.c_str();
    auto default_device = nnfusion::get_device_type(dev_name);
    if (default_device == CUDA_GPU || default_device == ROCM_GPU)
    {
        gpu_assign_thread_info(graph);
        naive_assign_stream_info(graph);
        assign_event_info(graph);
    }
    else if (default_device == GENERIC_CPU)
    {
        naive_assign_thread_info(graph);
        assign_event_info(graph);
    }
    else
    {
        throw nnfusion::errors::NotSupported("Not Supported Device Type");
    }

    NNFUSION_LOG(INFO) << "run async-------------------------------";
    return true;
}

// assign thread for cuda
void AssignAsyncInfoPass::gpu_assign_thread_info(shared_ptr<Graph>& graph)
{
    auto CPU_async_manager = AsyncManagerFactory::get_async_manager(graph, GENERIC_CPU);
    int num_device = FLAGS_fnum_device;
    if (num_device < 1)
        num_device = 1;

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

    if (num_async_node == 0 && num_device == 1)
    {
        for (auto gnode : graph->get_ordered_ops())
        {
            (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            int device_id = (*gnode)["DeviceID"].as<int>();
            async_info.execution_thread = CPU_async_manager->set_stream(device_id, "default");
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
            int device_id = (*gnode)["DeviceID"].as<int>();
            if (async_info.execution_thread == nullptr)
            {
                // constant, parameter and variable ops are in cuda_init(), and use default/main thread
                if (gnode->get_op_ptr()->is_tensor_op())
                {
                    async_info.execution_thread =
                        CPU_async_manager->set_stream(device_id, "default");
                }
                else if ((*gnode)["is_async_node"].is_valid() &&
                         (*gnode)["is_async_node"].as<bool>())
                {
                    async_info.execution_thread =
                        CPU_async_manager->set_stream(device_id, "async_" + to_string(count));
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
                        auto out_gnode = out_edge->get_dst();
                        int out_device_id = (*out_gnode)["DeviceID"].as<int>();
                        if (out_device_id == device_id)
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
                            auto out_gnode = out_edge->get_dst();
                            int out_device_id = (*out_gnode)["DeviceID"].as<int>();
                            if (out_device_id == device_id)
                                add_gnode(out_edge->get_dst());
                        }
                    }
                }
                else
                {
                    for (auto& in_edge : gnode->get_in_edges())
                    {
                        auto in_gnode = in_edge->get_src();
                        int in_device_id = (*in_gnode)["DeviceID"].as<int>();
                        if (in_device_id == device_id && !in_gnode->get_op_ptr()->is_tensor_op())
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
                        async_info.execution_thread =
                            CPU_async_manager->set_stream(device_id, "dev" + to_string(device_id));
                    }
                }
            }
        }
    }

    NNFUSION_LOG(INFO) << "assign thread info-------------------------------";
}

// assign thread of cpu
void AssignAsyncInfoPass::naive_assign_thread_info(shared_ptr<Graph>& graph)
{
    auto async_manager = AsyncManagerFactory::get_async_manager(graph, GENERIC_CPU);
    int n_stream = FLAGS_fnum_stream;
    if (n_stream < 0)
        n_stream = 1;
    if (n_stream == 1)
    {
        for (auto gnode : graph->get_ordered_ops())
        {
            if (!(*gnode)["Async_info"].is_valid())
                (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            int device_id = (*gnode)["DeviceID"].as<int>();
            async_info.execution_thread = async_manager->set_stream(device_id, "default");
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
            if (!(*gnode)["Async_info"].is_valid())
                (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            int device_id = (*gnode)["DeviceID"].as<int>();
            // all constant ops use default stream
            if (gnode->get_op_ptr()->is_tensor_op())
            {
                async_info.execution_thread = async_manager->set_stream(device_id, "default");
            }
            else
            {
                if (async_info.execution_thread == nullptr)
                {
                    async_info.execution_thread =
                        async_manager->set_stream(device_id, "base" + std::to_string(count));
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
    }

    // If enable runtime constant folding, ops whose inputs are all constants are taken as constant ops.
    // And will be called in init() instead of kernel_entry(). So these ops use default stream as well.

    if (FLAGS_frt_const_folding)
    {
        assign_default_info_for_const(graph, false);
    }

    NNFUSION_LOG(INFO) << "assign thread info-------------------------------";
}

// assign stream of gpu or thread of cpu
void AssignAsyncInfoPass::naive_assign_stream_info(shared_ptr<Graph>& graph)
{
    auto async_manager = AsyncManagerFactory::get_async_manager(graph, CUDA_GPU);
    bool allreduce_enable = FLAGS_fadd_allreduce;
    int n_stream = FLAGS_fnum_stream;
    if (n_stream < 0)
        n_stream = 1;
    if (n_stream == 1 && FLAGS_fuse_default_stream)
    {
        for (auto gnode : graph->get_ordered_ops())
        {
            if (!(*gnode)["Async_info"].is_valid())
                (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            int device_id = (*gnode)["DeviceID"].as<int>();
            auto thread = async_info.execution_thread;
            NNFUSION_CHECK(thread != nullptr);
            if (thread->is_default_stream())
            {
                async_info.execution_stream = async_manager->set_stream(device_id, "default");
            }
            else
            {
                std::string thread_symbol = thread->get_symbol() + "_thread_";
                async_info.execution_stream =
                    async_manager->set_stream(device_id, thread_symbol + "base");
            }
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
            if (!(*gnode)["Async_info"].is_valid())
                (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            int device_id = (*gnode)["DeviceID"].as<int>();
            // all constant ops use default stream
            if (gnode->get_op_ptr()->is_tensor_op())
            {
                auto thread = async_info.execution_thread;
                NNFUSION_CHECK(thread != nullptr);
                NNFUSION_CHECK(thread->is_default_stream());
                async_info.execution_stream = async_manager->set_stream(device_id, "default");
            }
            else
            {
                if (async_info.execution_stream == nullptr)
                {
                    auto thread = async_info.execution_thread;
                    NNFUSION_CHECK(thread != nullptr);
                    std::string thread_symbol = thread->get_symbol() + "_thread_";
                    async_info.execution_stream = async_manager->set_stream(
                        device_id, thread_symbol + "base" + std::to_string(count));
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
    }

    // If enable runtime constant folding, ops whose inputs are all constants are taken as constant ops.
    // And will be called in init() instead of kernel_entry(). So these ops use default stream as well.

    if (FLAGS_frt_const_folding)
    {
        assign_default_info_for_const(graph, true);
    }
    // add binding info
    for (auto gnode : graph->get_ordered_ops())
    {
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        auto kernel = get_kernel(gnode);
        if (auto kernel = std::dynamic_pointer_cast<CudaLibEmitter>(get_kernel(gnode)))
        {
            auto stream = async_info.execution_stream;
            if (kernel->require_cudnn_handle())
                stream->add_binding_symbol("cudnn_handle");
            if (kernel->require_cublas_handle())
                stream->add_binding_symbol("cublas_handle");
        }
    }
    NNFUSION_LOG(INFO) << "assign stream info-------------------------------";
}

// assign event of gpu or barrier of cpu
void AssignAsyncInfoPass::assign_event_info(std::shared_ptr<Graph>& graph)
{
    auto CPU_async_manager = AsyncManagerFactory::get_async_manager(graph, GENERIC_CPU);
    auto CUDA_async_manager = AsyncManagerFactory::get_async_manager(graph, CUDA_GPU);
    for (auto gnode : graph->get_ordered_ops())
    {
        NNFUSION_CHECK((*gnode)["Async_info"].is_valid());
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        auto stream = async_info.execution_stream;
        auto thread = async_info.execution_thread;

        for (auto& edge : gnode->get_in_edges())
        {
            auto input_gnode = edge->get_src();
            // constant and rt_const_folding ops are in xxx_init() of generated code,
            // so there is no need to add event.
            if (input_gnode->get_op_ptr()->is_tensor_op() ||
                (*input_gnode)["rt_const_folding"].is_valid_as<bool>())
            {
                continue;
            }
            auto& input_async_info = (*input_gnode)["Async_info"].as<AsyncExecutionInfo>();
            auto input_stream = input_async_info.execution_stream;
            auto input_thread = input_async_info.execution_thread;

            if (input_thread->get_stream_id() != thread->get_stream_id())
            {
                if (input_async_info.notify_barrier == nullptr)
                {
                    input_async_info.notify_barrier =
                        CPU_async_manager->set_event(input_thread, input_gnode->get_op_ptr());
                }
                async_info.wait_barriers.push_back(input_async_info.notify_barrier);
            }

            if (stream != nullptr && input_stream != nullptr)
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
                        input_async_info.record_event =
                            CUDA_async_manager->set_event(input_stream, input_gnode->get_op_ptr());
                    }
                    async_info.wait_events.push_back(input_async_info.record_event);
                }
            }
        }
    }
    NNFUSION_LOG(INFO) << "assign event info-------------------------------";
}

void AssignAsyncInfoPass::assign_default_info_for_const(std::shared_ptr<Graph>& graph,
                                                        bool assign_gpu_stream)
{
    NNFUSION_CHECK(FLAGS_frt_const_folding);
    auto CPU_async_manager = AsyncManagerFactory::get_async_manager(graph, GENERIC_CPU);
    auto CUDA_async_manager = AsyncManagerFactory::get_async_manager(graph, CUDA_GPU);
    set<string> constant_vals;
    for (auto gnode : graph->get_ordered_ops())
    {
        if (gnode->is_constant())
        {
            auto kernel = get_kernel(gnode);
            NNFUSION_CHECK_NOT_NULLPTR(kernel);
            (*gnode)["rt_const_folding"] = true;
            for (auto& out : kernel->m_context->output_names)
                constant_vals.insert(out);
        }
    }
    for (auto gnode : graph->get_ordered_ops())
    {
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        int device_id = (*gnode)["DeviceID"].as<int>();
        bool const_inputs = true;
        if (auto kernel = get_kernel(gnode))
        {
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
                (*gnode)["rt_const_folding"] = true;
                for (auto& out : kernel->m_context->output_names)
                {
                    constant_vals.insert(out);
                }
                async_info.execution_thread = CPU_async_manager->set_stream(device_id, "default");
                if (assign_gpu_stream)
                {
                    async_info.execution_stream =
                        CUDA_async_manager->set_stream(device_id, "default");
                }
            }
        }
    }
}

KernelEmitter::Pointer
    AssignAsyncInfoPass::get_kernel(std::shared_ptr<nnfusion::graph::GNode> gnode)
{
    KernelEmitter::Pointer kernel = nullptr;
    if (!gnode->is_parameter())
        NNFUSION_CHECK((*gnode)["Kernel_Selection_Result"].is_valid())
            << "Kernel should be selected before this pass:" << gnode->get_op_type();
    if ((*gnode)["Kernel_Selection_Result"].is_valid())
    {
        auto emitted_kernel = (*gnode)["Kernel_Selection_Result"]
                                  .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();

        if (!gnode->is_constant() && !emitted_kernel.second->is_emitted())
            NNFUSION_LOG(NNFUSION_WARNING) << "Kernel should be emitted before this pass:"
                                           << gnode->get_op_type();
        kernel = emitted_kernel.second;
    }

    return kernel;
}