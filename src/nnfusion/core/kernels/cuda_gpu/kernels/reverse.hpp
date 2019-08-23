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
            class Reverse : public CudaEmitter
            {
            public:
                Reverse(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                ngraph::Shape arg_shape;
                uint32_t arg_rank;
                ngraph::Shape result_shape;
                AxisSet reverse_axes;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion