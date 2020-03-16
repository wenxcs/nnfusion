// Microsoft (c) 2019, NNfusion Team
#pragma once

#include <string>

namespace nnfusion
{
    enum NNFusion_DeiveType
    {
        CUDA_GPU,
        ROCM_GPU,
        GENERIC_CPU,
        UNKNOWN
    };

    std::string get_device_str(NNFusion_DeiveType dt);
    NNFusion_DeiveType get_device_type(std::string dt);
}