// Microsoft (c) 2019, Wenxiang
#pragma once
#include "../cuda_function.hpp"
#include "../cuda_helper.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace cuda
    {
        class Convolution : public CudaFunction
        {
        public:
            ir::Convolution_p op;

        public:
            Convolution(ir::Operator_p inter_op);
            string codegen_test_name() override;

            LanguageUnit_p codegen_dependency() override;

            static CudaFunction_p codegen(ir::Operator_p inter_op);
        };

        using Convolution_p = shared_ptr<Convolution>;

        class ConvolutionCuda : public Convolution
        {
        public:
            ConvolutionCuda(ir::Operator_p inter_op);
            string codegen_function_name() override;

            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
            LanguageUnit_p codegen_dependency() override;
        };

        using ConvolutionCuda_p = shared_ptr<ConvolutionCuda>;

        class ConvolutionCudnn : public Convolution
        {
        public:
            ConvolutionCudnn(ir::Operator_p inter_op);
            string codegen_function_name() override;

            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
            LanguageUnit_p codegen_dependency() override;
        };

        using ConvolutionCudnn_p = shared_ptr<ConvolutionCudnn>;
    }
}