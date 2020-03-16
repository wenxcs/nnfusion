// Microsoft (c) 2019, NNfusion Team
#pragma once

#include <string>

namespace nnfusion
{
    enum NNFusion_DeviceType
    {
        CUDA_GPU,
        ROCM_GPU,
        GENERIC_CPU,
        UNKNOWN
    };

    std::string get_device_str(NNFusion_DeviceType dt);
    NNFusion_DeviceType get_device_type(std::string dt);
}