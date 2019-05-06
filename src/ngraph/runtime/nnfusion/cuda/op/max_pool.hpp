// Microsoft (c) 2019, Wenxiang
#pragma once
#include "../cuda_function.hpp"
#include "../cuda_helper.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace cuda
    {
        class MaxPool : public CudaFunction
        {
        public:
            ir::MaxPool_p op;

        public:
            MaxPool(ir::Operator_p inter_op);

            static CudaFunction_p codegen(ir::Operator_p inter_op);
        };

        using MaxPool_p = shared_ptr<MaxPool>;

        class MaxPool1D : public MaxPool
        {
        private:
            size_t window_width, window_stride, input_width, output_width;

        public:
            MaxPool1D(ir::Operator_p inter_op);
            string codegen_function_name() override;
            string codegen_test_name() override;

            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
            LanguageUnit_p codegen_dependency() override;
        };

        using MaxPool1D_p = shared_ptr<MaxPool1D>;

        class MaxPoolmD : public MaxPool
        {
        public:
            MaxPoolmD(ir::Operator_p inter_op)
                : MaxPool(inter_op)
            {
                enforce(op->padding_below == op->padding_above)
                    << "currently don't suport asymetric padding!";
            }
            string codegen_function_name() override;
            string codegen_test_name() override;

            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
            LanguageUnit_p codegen_dependency() override;
        };

        using MaxPool1D_p = shared_ptr<MaxPool1D>;
    }
}