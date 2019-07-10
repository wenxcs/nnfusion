// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "../cpu_kernel_emitter.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class Dot : public EigenKernelEmitter
            {
            public:
                Dot(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;

            private:
                shared_ptr<KernelContext> kernel_ctx;
                size_t reduction_axes;
                ngraph::Shape arg0_shape, arg1_shape;
                ngraph::Shape out_shape;
                ngraph::element::Type dtype;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion