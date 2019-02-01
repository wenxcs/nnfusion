// Microsoft (c) 2019, Wenxiang
#pragma once
#include "../cuda_function.hpp"
#include "../cuda_helper.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace cuda
    {
        class Reshape : public CudaFunction
        {
        public:
            ir::Reshape_p op;

        public:
            Reshape(ir::Operator_p inter_op);
            string codegen_test_name() override;
            LanguageUnit_p codegen_dependency() override;
            static CudaFunction_p codegen(ir::Operator_p inter_op);
        };

        using Reshape_p = shared_ptr<Reshape>;

        class Reshape2D : public Reshape
        {
        public:
            uint32_t block_size;
            NVShape input_strides;
            NVShape output_strides;
            NVShape trans_strides;

        public:
            Reshape2D(ir::Operator_p inter_op);
            string codegen_function_name() override;

            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
        };

        using Reshape2D_p = shared_ptr<Reshape2D>;

        class Reshape3D : public Reshape
        {
        public:
            std::vector<uint32_t> block_size;
            uint32_t block_size_x;
            NVShape input_strides, output_strides, trans_strides;

        public:
            Reshape3D(ir::Operator_p inter_op);
            string codegen_function_name() override;

            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
        };

        using Reshape3D_p = shared_ptr<Reshape3D>;

        class ReshapehD : public Reshape
        {
        public:
            uint32_t block_size_x;
            NVShape input_strides;
            NVShape output_strides;
            NVShape trans_strides;

        public:
            ReshapehD(ir::Operator_p inter_op);
            string codegen_function_name() override;

            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
        };

        using ReshapehD_p = shared_ptr<ReshapehD>;
    }
}