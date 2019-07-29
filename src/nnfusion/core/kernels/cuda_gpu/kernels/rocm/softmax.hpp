// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class RocmSoftmax : public CudaLibEmitter
            {
            public:
                RocmSoftmax(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;

            private:
                shared_ptr<KernelContext> kernel_ctx;
                ngraph::Shape input_shape, output_shape;
                ngraph::AxisSet axes;
                size_t height;
                size_t width;
                bool valid_inputs = true;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
