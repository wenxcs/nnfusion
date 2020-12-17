// Microsoft (c) 2019, NNFusion Team

#include "blockfusion_profiler.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion;
using namespace nnfusion::blockfusion;

void BlockFusionProfiler::set_profiling_context(
    BlockParallelDevice::Pointer _block_parallel_device,
    BlockFusionCudaCodegen::Pointer _blockfusion_codegen)
{
    if (_block_parallel_device != nullptr)
    {
        block_parallel_device = _block_parallel_device;
    }
    else
    {
        block_parallel_device = nullptr;
    }

    if (_blockfusion_codegen != nullptr)
    {
        blockfusion_codegen = _blockfusion_codegen;
    }
    else
    {
        blockfusion_codegen = nullptr;
    }
}

blockfusion::ProfilingResult BlockFusionProfiler::get_codegen_profiling_result(
    BlockFusionCudaCodegen::Pointer codegen_context)
{
    NNFUSION_CHECK_NOT_NULLPTR(codegen_context);
    auto kernel = std::dynamic_pointer_cast<KernelEmitter>(codegen_context);
    NNFUSION_CHECK_NOT_NULLPTR(kernel);

    blockfusion::ProfilingResult result;
    result.profile_codegen = true;

    NNFUSION_LOG(INFO) << "Profiling BlockFusionCudaCodegen";

    // profiling fused_execution_time, enable by uncommenting the code block
    /*nnfusion::profiler::IProfilingRuntime::Pointer runtime =
        nnfusion::profiler::get_default_runtime(DeviceType::CUDA_GPU);
    nnfusion::profiler::ProfilingContext::Pointer pctx =
        std::make_shared<nnfusion::profiler::ProfilingContext>(kernel);
    nnfusion::profiler::Profiler prof(runtime, pctx);
    if (!prof.execute())
        NNFUSION_LOG(INFO) << "Kernel Failed.";
    else
    {
        double fused_execution_time = pctx->result.get_device_avg() * 1000;
        NNFUSION_LOG(INFO) << codegen_context->get_function_name()
                  << " time cost(us):" << fused_execution_time;
        result.fused_execution_time = fused_execution_time;
    }*/

    result.num_parameters = kernel->m_context->inputs.size() + kernel->m_context->outputs.size() +
                            kernel->m_context->tensors.size();

    return result;
}

blockfusion::ProfilingResult BlockFusionProfiler::get_profiling_result()
{
    blockfusion::ProfilingResult profiling_result;

    if (block_parallel_device != nullptr)
    {
        auto device_profiling = block_parallel_device->get_profiling_result();

        profiling_result.profile_device = true;
        profiling_result.num_bes = device_profiling.num_bes;
        profiling_result.num_kernels = device_profiling.num_kernels;
        profiling_result.num_large_kernels = device_profiling.num_large_kernels;
        profiling_result.normal_execution_time = device_profiling.normal_execution_time;
        profiling_result.fused_estimation_time = device_profiling.fused_estimation_time;
    }
    else
    {
        profiling_result.profile_device = false;
        profiling_result.num_bes = 0;
        profiling_result.num_kernels = 0;
        profiling_result.num_large_kernels = 0;
        profiling_result.normal_execution_time = 0;
        profiling_result.fused_estimation_time = 0;
    }

    if (blockfusion_codegen != nullptr)
    {
        auto codegen_profiling = get_codegen_profiling_result(this->blockfusion_codegen);

        profiling_result.profile_codegen = true;
        profiling_result.num_parameters = codegen_profiling.num_parameters;
        profiling_result.fused_execution_time = codegen_profiling.fused_execution_time;
    }
    else
    {
        profiling_result.profile_codegen = false;
        profiling_result.num_parameters = 0;
        profiling_result.fused_execution_time = 0;
    }

    return profiling_result;
}
