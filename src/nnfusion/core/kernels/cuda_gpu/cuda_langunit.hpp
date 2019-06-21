// Microsoft (c) 2019, Wenxiang
#pragma once
#include "nnfusion/core/kernels/common_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace header
        {
            LU_DECLARE(cuda);
            LU_DECLARE(cublas);
            LU_DECLARE(cudnn);
        }

        namespace macro
        {
            LU_DECLARE(CUDA_SAFE_CALL_NO_THROW);
            LU_DECLARE(CUDA_SAFE_CALL);
            LU_DECLARE(CUDNN_SAFE_CALL_NO_THROW);
            LU_DECLARE(CUDNN_SAFE_CALL);
            LU_DECLARE(CUBLAS_SAFE_CALL_NO_THROW);
            LU_DECLARE(CUBLAS_SAFE_CALL);
            LU_DECLARE(CUDA_SAFE_LAUNCH);
        }

        namespace declaration
        {
            LU_DECLARE(division_by_invariant_multiplication);
            LU_DECLARE(load);
            LU_DECLARE(mad16);
            LU_DECLARE(mod16);
            LU_DECLARE(global_cublas_handle);
            LU_DECLARE(global_cudnn_handle);
            LU_DECLARE(num_SMs);
        }
    } // namespace kernels
} // namespace nnfusion