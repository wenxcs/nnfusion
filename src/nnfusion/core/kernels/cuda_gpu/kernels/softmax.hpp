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
            class Softmax : public CudaEmitter
            {
            public:
                Softmax(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                shared_ptr<KernelContext> kernel_ctx;
                ngraph::Shape input_shape, output_shape;
                ngraph::AxisSet axes;
                size_t expected_block_size;
                size_t height;
                size_t width;
                bool valid_inputs = true;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion