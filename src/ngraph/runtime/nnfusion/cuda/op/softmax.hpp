// Microsoft (c) 2019, Yuchao
#pragma once
#include "../cuda_function.hpp"
#include "../cuda_helper.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace cuda
    {
        class Softmax : public CudaFunction
        {
        public:
            ir::Softmax_p op;
            size_t out_rank, reduce_rank, non_reduce_rank, rank, nthreads;
            std::vector<int> reduce_strides_magic;
            std::vector<int> reduce_strides_shift;
            std::vector<int> non_reduce_strides_magic;
            std::vector<int> non_reduce_strides_shift;
            ngraph::NVShape non_reduce_shape;
            ngraph::NVShape non_reduce_strides;
            ngraph::NVShape non_reduce_strides_in_input;
            ngraph::NVShape reduce_shape;
            ngraph::NVShape reduce_strides;
            ngraph::NVShape reduce_strides_in_input;
            size_t reduce_count;

            Softmax(ir::Operator_p inter_op);
            string codegen_test_name() override;

            static CudaFunction_p codegen(ir::Operator_p inter_op);
        };
        using Softmax_p = shared_ptr<Softmax>;

        class SoftmaxStridesBackOne : public Softmax
        {
        public:
            SoftmaxStridesBackOne(ir::Operator_p inter_op);
            string codegen_function_name() override;

            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
            LanguageUnit_p codegen_dependency() override;
        };
        using SoftmaxStridesBackOne_p = shared_ptr<SoftmaxStridesBackOne>;

        class SoftmaxStridesBackNotOne : public Softmax
        {
        public:
            SoftmaxStridesBackNotOne(ir::Operator_p inter_op);
            string codegen_function_name() override;

            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
            LanguageUnit_p codegen_dependency() override;
        };
        using SoftmaxStridesBackNotOne_p = shared_ptr<SoftmaxStridesBackNotOne>;
    }
}
