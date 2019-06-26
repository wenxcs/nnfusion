// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "nnfusion/core/kernels/common_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace header
        {
            LU_DECLARE(thread);
            LU_DECLARE(eigen_tensor);
            LU_DECLARE(eigen_utils);
            LU_DECLARE(cblas);
        }

        namespace macro
        {
        }

        namespace declaration
        {
            LU_DECLARE(eigen_global_thread_pool);
            LU_DECLARE(eigen_global_thread_pool_device);
            LU_DECLARE(cblas_sgemm_batch);
        }
    } // namespace kernels
} // namespace nnfusion