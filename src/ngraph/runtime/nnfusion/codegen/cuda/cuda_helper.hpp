
// Microsoft (c) 2019, Wenxiang
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_errorcheck.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_kernelops.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_testutil.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            namespace codegen
            {
                namespace cuda
                {
                    void get_math_kernel(CodeWriter& writer,
                                         const std::string& name,
                                         const std::string& math_kernel,
                                         const std::vector<std::string>& data_types);

                    uint32_t align_to_block_size(uint32_t threads, uint32_t block_size);

                    void emit_memcpyDtD(CodeWriter& writer,
                                        const TensorWrapper& dst,
                                        const TensorWrapper& src,
                                        size_t buffer_size = 0);
                }
            }
        }
    }
}