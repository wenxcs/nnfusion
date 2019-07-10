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
            class ConvolutionCudnn : public CudaLibEmitter
            {
            public:
                ConvolutionCudnn(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;

            private:
                ngraph::Shape input_shape, filter_shape, output_shape;
                ngraph::Strides window_dilation_strides, window_movement_strides,
                    data_dilation_strides;
                ngraph::CoordinateDiff padding_below_diff, padding_above_diff;
                string dtype;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion