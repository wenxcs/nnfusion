// Microsoft (c) 2019, Wenxiang
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "../core/codegenerator.hpp"
#include "../core/common.hpp"
#include "../core/languageunit.hpp"
#include "../core/op.hpp"
#include "../op/alist.hpp"
#include "cuda_errorcheck.hpp"
#include "cuda_kernelops.hpp"
#include "cuda_testutil.hpp"

using CodeWriter = ngraph::codegen::CodeWriter;

namespace nnfusion
{
    namespace cuda
    {
        shared_ptr<LanguageUnit> get_math_kernel(const std::string& name,
                                                 const std::string& math_kernel,
                                                 const std::vector<std::string>& data_types);

        uint32_t align_to_block_size(uint32_t threads, uint32_t block_size);

        void emit_memcpyDtD(CodeWriter& writer,
                            const TensorWrapper& dst,
                            const TensorWrapper& src,
                            size_t buffer_size = 0);

        void coordinate_transform_to_multi_d(CodeWriter& writer,
                                             std::string i_strides,
                                             std::string i_stride_magic,
                                             std::string i_stride_shift,
                                             std::string i_coord_product,
                                             std::string o_coordinates,
                                             size_t rank,
                                             bool register_arguments);

        std::string collective_coordinate_transform_helper(CodeWriter& writer,
                                                           std::string i_thread_index,
                                                           std::string i_strides,
                                                           std::string i_stride_magic,
                                                           std::string i_stride_shift,
                                                           std::string i_reduced_strides,
                                                           std::string o_coordinates,
                                                           size_t rank,
                                                           bool register_arguments);
    }
}
