// Microsoft (c) 2019, MSRA/NNFUSION Team
/**
 * \brief Use this Profiler to run each operator
 * \author wenxh
 * \todo This profiler only support linux since it will invoke native commands.
 */

#include "profiler.hpp"
#include "nnfusion/core/graph/graph_util.hpp"

#include <chrono>
#include <ctime>
#include <ratio>

using namespace ngraph;
using namespace nnfusion;
using namespace nnfusion::profiler;
using namespace std::chrono;

Profiler::Profiler(IProfilingRuntime::Pointer rt, ProfilingContext::Pointer context)
{
    this->rt = rt;
    this->pctx = context;
    //\todo: verify if the runtime is ok
}

double Profiler::execute(void** input, void** output)
{
    if (rt == nullptr)
        return -1.0;

    for (int i = 0; i < pctx->host_times; i++)
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        double device_time_span = rt->execute(this->pctx, input, output);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        if (device_time_span < 0)
        {
            LOG_WARN << "Kernel launch failed.";
            continue;
        }
        pctx->result.record_host_duration(time_span.count());
        pctx->result.record_device_duration(device_time_span);
    }
    pctx->result.set_ready();
    return pctx->result.get_device_avg();
}

bool Profiler::execute()
{
    auto& kernel_mem = pctx->kernel_memory;
    bool ret = execute(kernel_mem->unsafe_inputs(), kernel_mem->unsafe_outputs()) >= 0;
    return ret;
}

bool Profiler::find_best()
{
    return false;
}

bool Profiler::execute_all()
{
    return false;
}

void GraphEvaluate::create_profiling_contexts(shared_ptr<GNode> gnode)
{
    auto node = gnode->get_op_ptr();
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(
            node->description(), GENERIC_CPU, DT_FLOAT);
    shared_ptr<KernelContext> ctx(new KernelContext(node));

    for (auto kernel_reg : kernel_regs)
    {
        if (kernel_reg->m_tag != "reference")
            continue;
        auto kernel = kernel_reg->m_factory(ctx);
        if (kernel->emit_source())
        {
            // Replacing the kernel;
            auto pctx = make_shared<ProfilingContext>(kernel);
            this->gctx.set_profiling_context(gnode, pctx);
            return;
        }
    }

    LOG_ERR << "Invalid reference kenel for " << gnode->get_name() << ".";
}

void GraphEvaluate::connect_nodes(shared_ptr<GNode> gnode)
{
    auto pctx = gctx.get_profiling_context(gnode);
    for (auto& edge : gnode->get_out_edges())
    {
        // Skip control edge
        if (edge->is_control_edge())
            continue;
        auto dstnode = edge->get_dst();
        auto dstpctx = gctx.get_profiling_context(dstnode);
        // This statments will remove some allocated memory.
        pctx->kernel_memory->forward(
            edge->get_src_output(), dstpctx->kernel_memory, edge->get_dst_input());
    }
}

unordered_map<string, ProfilingContext::Pointer> GraphEvaluate::eval()
{
    auto ordered_ops = gctx.graph->get_ordered_ops();
    for (auto& op : ordered_ops)
        create_profiling_contexts(op);
    for (auto& op : ordered_ops)
        connect_nodes(op);
    for (auto& node : ordered_ops)
    {
        auto pctx = gctx.get_profiling_context(node);
        // Constant
        if (node->is_constant())
        {
            auto const_node = static_pointer_cast<op::Constant>(node->get_op_ptr());
            pctx->kernel_memory->set_output_from(
                0, const_node->get_data_ptr(), const_node->get_data_size());
        }
        else
        {
            rt->execute(
                pctx, pctx->kernel_memory->unsafe_inputs(), pctx->kernel_memory->unsafe_outputs());
        }
    }

    unordered_map<string, ProfilingContext::Pointer> result;
    for (auto& outnode : gctx.graph->get_outputs())
        result[outnode->get_unique_name()] = gctx.get_profiling_context(outnode);
    // The result data ptr is like result["nodename"]->kernel_memory->unsafe_output(0);
    return move(result);
}