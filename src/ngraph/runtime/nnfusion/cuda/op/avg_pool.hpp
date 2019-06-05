// Microsoft (c) 2019, Wenxiang
#pragma once
#include "../cuda_function.hpp"
#include "../cuda_helper.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace cuda
    {
        struct pooling_op_shape
        {
            int N;
            int C;
            int D;
            int H;
            int W;
            int K;
            int M;
            int P;
            int Q;
            int J;
            int T;
            int R;
            int S;
            int STRIDE_D;
            int STRIDE_H;
            int STRIDE_W;
            int PAD_D;
            int PAD_H;
            int PAD_W;
        };

        class AvgPool : public CudaFunction
        {
        public:
            ir::AvgPool_p op;

        public:
            AvgPool(ir::Operator_p inter_op);
            static pooling_op_shape avgpool_shape(
                NVShape in, NVShape out, NVShape window, NVShape strides, NVShape pad);
            static CudaFunction_p codegen(ir::Operator_p inter_op);
        };

        using AvgPool_p = shared_ptr<AvgPool>;

        class AvgPool1D : public AvgPool
        {
        public:
            // precompute for fast constant memory access
            int HW, DHW, CDHW, PQ, MPQ, KMPQ, RS, TRS;
            int magic_N, shift_N, magic_P, shift_P, shift_S, magic_S, magic_RS, shift_RS;
            float alpha, beta;
            pooling_op_shape shape;

        public:
            AvgPool1D(ir::Operator_p inter_op);
            string codegen_function_name() override;
            string codegen_test_name() override;

            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
            LanguageUnit_p codegen_dependency() override;
        };

        using AvgPool1D_p = shared_ptr<AvgPool1D>;

        // Support 2d & 3d Padding
        class AvgPoolmD : public AvgPool
        {
        public:
            AvgPoolmD(ir::Operator_p inter_op)
                : AvgPool(inter_op)
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

        using AvgPool1D_p = shared_ptr<AvgPool1D>;
    }
}