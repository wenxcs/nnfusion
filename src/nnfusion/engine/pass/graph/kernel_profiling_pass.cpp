
// Microsoft (c) 2019, NNFusion Team
#include "kernel_profiling_pass.hpp"
#include "nnfusion/engine/profiler/cuda_runtime.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::profiler;
DEFINE_bool(fenable_kernel_profiling, false, "profile kernel time.");
DECLARE_string(fstream_assign_policy);

bool KernelProfilingPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (!FLAGS_fenable_kernel_profiling && FLAGS_fstream_assign_policy != "kernel_prof_based")
        return true;

    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    for (auto it : nodes)
    {
        if ((*it)["Kernel_Selection_Result"].is_valid())
        {
            auto kernel_result = (*it)["Kernel_Selection_Result"]
                                     .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
            auto n_device_type = kernel_result.first;
            auto kernel = kernel_result.second;

            if (!(*it)["Kernel_Profiling_Result"].is_valid() && !it->is_constant())
            {
                auto profiling_kernel =
                    [](KernelEmitter::Pointer kernel,
                       IProfilingRuntime::Pointer runtime) -> KernelProfilingRecord::Pointer {
                    if (kernel->get_or_emit_source())
                    {
                        auto pctx = make_shared<nnfusion::profiler::ProfilingContext>(kernel);
                        nnfusion::profiler::Profiler prof(runtime, pctx);

                        if (prof.execute())
                        {
                            double kernel_time = pctx->result.get_device_avg();
                            auto record = make_shared<KernelProfilingRecord>();
                            record->kernel_time_in_us = kernel_time;
                            record->valid = true;

                            NNFUSION_LOG(INFO)
                                << "Profiling kernel: " << kernel->get_function_name()
                                << ", kernel time(us):" << kernel_time;
                            return record;
                        }
                        else
                        {
                            NNFUSION_LOG(INFO) << "Kernel Failed.";
                        }
                    }
                    return nullptr;
                };
                KernelProfilingRecord::Pointer result;

                if (n_device_type == CUDA_GPU)
                {
                    result = profiling_kernel(kernel, CUPTIRuntime::Runtime());
                }
                else if (n_device_type == GENERIC_CPU)
                {
                    result = profiling_kernel(kernel, CPUDefaultRuntime::Runtime());
                }
                else
                {
                    result = profiling_kernel(kernel, get_default_runtime(n_device_type));
                    if (!result && n_device_type == ROCM_GPU)
                    {
                        result = profiling_kernel(kernel, get_default_runtime(CUDA_GPU));
                    }
                }

                if (result)
                {
                    (*it)["Kernel_Profiling_Result"] = result;
                }
            }
        }
    }

    return true;
}