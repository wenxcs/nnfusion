// Microsoft (c) 2019, Wenxiang
#pragma once
#include "../cuda_function.hpp"
#include "../cuda_helper.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace cuda
    {
        class Concat : public CudaFunction
        {
        public:
            ir::Concat_p op;
            size_t input_num, split_input_size, residue, concat_axis, split_input_stride_offset;
            std::vector<uint32_t> inputs_strides, split_nthreads, split_output_strides,
                split_input_stride_offsets, split_aligned_grid_size_x;
            uint32_t output_stride;
            uint32_t block_size_x;

        public:
            Concat(ir::Operator_p inter_op);
            string codegen_function_name() override;
            string codegen_test_name() override;

            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
            LanguageUnit_p codegen_dependency() override;

            static CudaFunction_p codegen(ir::Operator_p inter_op);
        };

        using Concat_p = shared_ptr<Concat>;
    }
}