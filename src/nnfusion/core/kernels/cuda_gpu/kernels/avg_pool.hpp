// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
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

            class AvgPool1D : public BlockCudaEmitter
            {
            public:
                AvgPool1D(shared_ptr<KernelContext> ctx);
                static pooling_op_shape avgpool_shape(ngraph::Shape in,
                                                      ngraph::Shape out,
                                                      ngraph::Shape window,
                                                      ngraph::Shape strides,
                                                      ngraph::Shape pad);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                shared_ptr<KernelContext> kernel_ctx;
                ngraph::Shape input_shape, output_shape, window_shape, padding_below, padding_above;
                ngraph::Strides window_stride;
                bool include_pad;
                string input_type, output_type;

                // Precompute for fast constant memory access.
                int HW, DHW, CDHW, PQ, MPQ, KMPQ, RS, TRS;
                int magic_N, shift_N, magic_P, shift_P, shift_S, magic_S, magic_RS, shift_RS;
                float alpha, beta;
                pooling_op_shape shape;
            };

            class AvgPoolmD : public CudaLibEmitter
            {
            public:
                AvgPoolmD(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;

            private:
                shared_ptr<KernelContext> kernel_ctx;
                ngraph::Shape input_shape, output_shape, window_shape, padding_below, padding_above;
                ngraph::Strides window_stride;
                bool include_pad;
                string input_type, output_type;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion