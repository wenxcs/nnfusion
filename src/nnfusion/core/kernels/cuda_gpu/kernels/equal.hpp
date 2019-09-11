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
            class Equal : public CudaEmitter
            {
            public:
                Equal(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                ngraph::Shape input_shape_0, input_shape_1, output_shape;
                int64_t equal_size;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion