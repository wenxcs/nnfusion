// Microsoft (c) 2019, Wenxiang Hu
#include "kernel_selection.hpp"

#include <queue>
#include <utility>

using namespace nnfusion;
using namespace nnfusion::profiler;

// Register_Tag(Enable_Kernel_Selection, bool);
// Register_Tag(Kernel_Selection_Device, DeviceType);
// Register_Tag(Kernel_Selection_Result, vector<pair<DeviceType, KernelEmitter>>);

pair<DeviceType, kernels::KernelEmitter::Pointer> ProfilingBasedKernelSelector::profiling_best(
    shared_ptr<ngraph::Node> node, DeviceType devtype, IProfilingRuntime::Pointer runtime)
{
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(node->description(), devtype, DT_FLOAT);

    // Skip since only one candidate
    if (kernel_regs.size() == 1)
        return std::make_pair(devtype, nullptr);

    shared_ptr<KernelContext> ctx(new KernelContext(node));

    bool has_valid_kernel = false;
    LOG_INFO << "Start profiling...";
    auto comparef = [](const ProfilingContext::Pointer& a, const ProfilingContext::Pointer& b) {
        return a->result.get_device_avg() > b->result.get_device_avg();
    };
    priority_queue<ProfilingContext::Pointer, vector<ProfilingContext::Pointer>, decltype(comparef)>
        prof_res(comparef);
    for (auto kernel_reg : kernel_regs)
    {
        auto kernel = kernel_reg->m_factory(ctx);
        if (kernel->get_or_emit_source())
        {
            has_valid_kernel = true;
            auto pctx = make_shared<nnfusion::profiler::ProfilingContext>(kernel);
            nnfusion::profiler::Profiler prof(runtime, pctx);

            if (!prof.execute())
                LOG_INFO << "Kernel Failed.";
            else
            {
                LOG_INFO << "Kernel Emitter#" << prof_res.size()
                         << " time cost(ms):" << pctx->result.get_device_avg();
                prof_res.push(pctx);
            }
        }
    }

    while (!prof_res.empty())
    {
        auto best = prof_res.top();
        prof_res.pop();
        ///\todo Check if the result is ready.
        if (!best->result.is_ready())
            continue;
        LOG_INFO << "Best kernel time cost(ms):" << best->result.get_device_avg();
        return std::make_pair(devtype, move(best->kernel));
    }
    return std::make_pair(devtype, nullptr);
}

bool ProfilingBasedKernelSelector::run(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu)
{
    bool enable_tuning =
        getenv("NNFUSION_ENABLE_TUNING") ? bool(atoi(getenv("NNFUSION_ENABLE_TUNING"))) : true;
    if (!enable_tuning)
        return true;

    // Config area
    vector<string> white_list{"Broadcast"};
    bool all_device = false;
    DeviceType the_device = ROCM_GPU;

    // Currently *ONLY* has BroadCast Selection
    auto& p = tu->program;
    for (auto iterator : p)
    {
        for (auto ins : *iterator)
        {
            auto opname = ins->operatorDef()->get_name();
            for (auto& rule : white_list)
                if (opname.find(rule) < opname.size())
                {
                    (*ins)["Enable_Kernel_Selection"] = true;
                    if (!all_device)
                        (*ins)["Kernel_Selection_Device"] = the_device;
                }
        }
    }

    for (auto iterator : p)
    {
        for (auto ins : *iterator)
        {
            if ((*ins)["Enable_Kernel_Selection"].is_valid() &&
                (*ins)["Enable_Kernel_Selection"].as<bool>())
            {
                vector<pair<DeviceType, KernelEmitter::Pointer>> res;
                if (!(*ins)["Kernel_Selection_Device"].is_valid() ||
                    ((*ins)["Kernel_Selection_Device"].as<DeviceType>() == CUDA_GPU))
                {
                    auto rt =
                        dynamic_pointer_cast<IProfilingRuntime>(CudaDefaultRuntime::Runtime());
                    auto ans = profiling_best(ins->operatorDef(), CUDA_GPU, rt);
                    if (ans.second != nullptr)
                        res.push_back(ans);
                }

                if (!(*ins)["Kernel_Selection_Device"].is_valid() ||
                    ((*ins)["Kernel_Selection_Device"].as<DeviceType>() == ROCM_GPU))
                {
                    if (RocmDefaultRuntime::Runtime()->check_env())
                    {
                        auto rt =
                            dynamic_pointer_cast<IProfilingRuntime>(RocmDefaultRuntime::Runtime());
                        auto ans = profiling_best(ins->operatorDef(), ROCM_GPU, rt);
                        if (ans.second != nullptr)
                            res.push_back(ans);
                    }
                    else
                        LOG_WARN << "Rocm runtime is not available.";
                }

                if (!(*ins)["Kernel_Selection_Device"].is_valid() ||
                    ((*ins)["Kernel_Selection_Device"].as<DeviceType>() == GENERIC_CPU))
                {
                    if (false)
                    {
                        auto rt =
                            dynamic_pointer_cast<IProfilingRuntime>(ReferenceRuntime::Runtime());
                        auto ans = profiling_best(ins->operatorDef(), GENERIC_CPU, rt);
                        if (ans.second != nullptr)
                            res.push_back(ans);
                    }
                    else
                        LOG_WARN << "CPU runtime is not available.";
                }

                (*ins)["Kernel_Selection_Result"] = move(res);
            }
        }
    }
    return true;
}