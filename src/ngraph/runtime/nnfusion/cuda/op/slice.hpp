// Microsoft (c) 2019, Wenxiang
#pragma once
#include "../cuda_function.hpp"
#include "../cuda_helper.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace cuda
    {
        class Slice : public CudaFunction
        {
        public:
            ir::Slice_p op;
            uint32_t nthreads;
            uint32_t block_size_x;
            uint32_t aligned_grid_size_x;
            NVShape output_strides;
            NVShape input_strides;

        public:
            Slice(ir::Operator_p inter_op);
            string codegen_function_name() override;
            string codegen_test_name() override;

            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
            LanguageUnit_p codegen_dependency() override;

            static CudaFunction_p codegen(ir::Operator_p inter_op);
        };

        using Slice_p = shared_ptr<Slice>;
    }
}