// Microsoft (c) 2019, Wenxiang Hu
#include "kernel_selection.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

#include <queue>

using namespace nnfusion;
using namespace nnfusion::profiler;

bool ProfilingBasedKernelSelector::run(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu)
{
    bool enable_tuning =
        getenv("NNFUSION_ENABLE_TUNING") ? bool(atoi(getenv("NNFUSION_ENABLE_TUNING"))) : true;
    if (!enable_tuning)
        return true;

    // Currently *ONLY* has BroadCast Selection
    auto& p = tu->program;
    for (auto iterator = p.entry; iterator != nullptr; iterator = iterator->next)
    {
        for (auto ins : *iterator)
        {
            auto opname = ins->operatorDef()->get_name();
            if (opname.find("Broadcast") < opname.size())
            {
                ins->Tag().Set<bool>("rocm_prof_enable", true);
            }
        }
    }

    for (auto iterator = p.entry; iterator != nullptr; iterator = iterator->next)
    {
        for (auto ins : *iterator)
        {
            if (ins->Tag().hasAttribute(
                    "rocm_prof_enable")) // && ins->Tag().hasAttribute("Device"))
            {
                auto node = ins->operatorDef();
                if (!RocmDefaultRuntime::Runtime()->check_env())
                {
                    LOG_WARN << "Does not have ROCM runtime.";
                    continue;
                }
                std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
                    KernelRegistry::Global()->FindKernelRegistrations(
                        node->description(), DeviceType::ROCM_GPU, DT_FLOAT);
                shared_ptr<KernelContext> ctx(new KernelContext(node));

                bool has_valid_kernel = false;
                LOG_INFO << "Start profiling...";
                auto comparef = [](const ProfilingContext::Pointer& a,
                                   const ProfilingContext::Pointer& b) {
                    return a->result.get_device_avg() > b->result.get_device_avg();
                };
                priority_queue<ProfilingContext::Pointer,
                               vector<ProfilingContext::Pointer>,
                               decltype(comparef)>
                    prof_res(comparef);
                for (auto kernel_reg : kernel_regs)
                {
                    auto kernel = kernel_reg->m_factory(ctx);
                    if (kernel->emit_source())
                    {
                        has_valid_kernel = true;
                        auto pctx = make_shared<nnfusion::profiler::ProfilingContext>(kernel);
                        nnfusion::profiler::Profiler prof(
                            nnfusion::profiler::RocmDefaultRuntime::Runtime(), pctx);

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
                    //\todo Check if the result is ready.
                    if (!best->result.is_ready())
                        continue;
                    LOG_INFO << "Best kernel time cost(ms):" << best->result.get_device_avg();
                    ins->Tag().Set<KernelEmitter::Pointer>("KernelCandidate", move(best->kernel));
                    ins->Tag().Set<DeviceType>("KernelCandidateDevice", DeviceType::ROCM_GPU);
                    break;
                }
            }
        }
    }
    return true;
}