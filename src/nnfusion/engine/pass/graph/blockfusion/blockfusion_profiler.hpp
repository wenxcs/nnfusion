// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "block_parallel_device.hpp"
#include "blockfusion_codegen.hpp"
#include "common.hpp"

using namespace nnfusion;
using namespace nnfusion::blockfusion;

class BlockFusionProfiler
{
public:
    BlockFusionProfiler(BlockParallelDevice::Pointer _block_parallel_device = nullptr,
                        BlockFusionCudaCodegen::Pointer _blockfusion_codegen = nullptr)
    {
        set_profiling_context(_block_parallel_device, _blockfusion_codegen);
    }

    void set_profiling_context(BlockParallelDevice::Pointer _block_parallel_device = nullptr,
                               BlockFusionCudaCodegen::Pointer _blockfusion_codegen = nullptr);

    blockfusion::ProfilingResult get_profiling_result();

private:
    blockfusion::ProfilingResult
        get_codegen_profiling_result(BlockFusionCudaCodegen::Pointer codegen_context);

private:
    BlockParallelDevice::Pointer block_parallel_device;
    BlockFusionCudaCodegen::Pointer blockfusion_codegen;
};