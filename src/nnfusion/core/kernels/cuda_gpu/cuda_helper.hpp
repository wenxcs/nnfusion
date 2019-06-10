// Microsoft (c) 2019, Wenxiang
#pragma once

#include "nnfusion/core/kernels/common.hpp"
#include "nnfusion/core/kernels/cuda_gpu/util/gpu_util.hpp"
#include "nnfusion/core/kernels/cuda_gpu/util/nvshape.hpp"
#include "nnfusion/core/kernels/language_unit.hpp"

#include "cuda_kernelops.hpp"
#include "cuda_testutil.hpp"

using CodeWriter = ngraph::codegen::CodeWriter;

namespace nnfusion
{
    namespace kernels
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

            std::string
                collective_coordinate_transform_helper(CodeWriter& writer,
                                                       std::string i_thread_index,
                                                       std::string i_strides,
                                                       std::string i_stride_magic,
                                                       std::string i_stride_shift,
                                                       std::string i_reduced_strides,
                                                       std::string o_coordinates,
                                                       size_t rank,
                                                       bool register_arguments,
                                                       std::string reduced_idx = "reduced_idx");

            void div_to_mul(const ngraph::NVShape& shape,
                            std::vector<int>& magic,
                            std::vector<int>& shift);

            void get_reduce_strides(NVShape input_shape,
                                    NVShape reduce_axis,
                                    NVShape& non_reduce_shape,
                                    NVShape& non_reduce_strides,
                                    NVShape& non_reduce_strides_in_input,
                                    NVShape& reduce_shape,
                                    NVShape& reduce_strides,
                                    NVShape& reduce_strides_in_input);

            void simplify_reduce_shape(NVShape in,
                                       NVShape reduce_axis,
                                       NVShape& simplified_shape,
                                       NVShape& simplified_reduce_axis);
        }
    }
}
