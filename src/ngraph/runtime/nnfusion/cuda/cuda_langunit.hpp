// Microsoft (c) 2019, Wenxiang
#pragma once
#include "../core/languageunit.hpp"

#define LU_DEFINE(NAME) extern LanguageUnit_p NAME;

namespace nnfusion
{
    namespace cuda
    {
        namespace header
        {
            LU_DEFINE(stdio);
            LU_DEFINE(fstream);
            LU_DEFINE(stdexcept);
            LU_DEFINE(sstream);
            LU_DEFINE(cuda);
            LU_DEFINE(cublas);
            LU_DEFINE(cudnn);
            LU_DEFINE(assert);
        }

        namespace macro
        {
            LU_DEFINE(NNFUSION_DEBUG);
            LU_DEFINE(CUDA_SAFE_CALL_NO_THROW);
            LU_DEFINE(CUDA_SAFE_CALL);
            LU_DEFINE(CUDNN_SAFE_CALL_NO_THROW);
            LU_DEFINE(CUDNN_SAFE_CALL);
            LU_DEFINE(CUBLAS_SAFE_CALL_NO_THROW);
            LU_DEFINE(CUBLAS_SAFE_CALL);
        }

        namespace declaration
        {
            LU_DEFINE(typedef_int);
            LU_DEFINE(division_by_invariant_multiplication);
            LU_DEFINE(load);
            LU_DEFINE(global_cublas_handle);
            LU_DEFINE(global_cudnn_handle);
        }
    }
}

#undef LU_DEFINE