// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "../cpu_kernel_emitter.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class AvgPoolMlas : public MlasKernelEmitter
            {
            public:
                AvgPoolMlas(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;

            private:
                ngraph::Shape input_shape, output_shape, window_shape, padding;
                ngraph::Shape padding_below, padding_above;
                ngraph::Strides window_stride;
                bool include_pad;
                string dtype;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
