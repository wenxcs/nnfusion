// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "cuda_helper.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            struct dim3
            {
                dim3()
                    : x(1)
                    , y(1)
                    , z(1)
                {
                }
                dim3(int x, int y, int z)
                    : x(x)
                    , y(y)
                    , z(z)
                {
                }
                int x, y, z;
            };

            class CudaEmitter : public KernelEmitter
            {
            public:
                CudaEmitter(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cuda")
                {
                }

                FunctionUnit_p get_or_emit_source() override;

                virtual bool is_static_function() override { return false; }
            protected:
                // config the blockDim and gridDim
                virtual void set_launch_config() = 0;

                LanguageUnit_p emit_function_call() override;
                LanguageUnit_p emit_function_signature() override;

                dim3 m_blockDim;
                dim3 m_gridDim;
            };

            class CudaLibEmitter : public KernelEmitter
            {
            public:
                CudaLibEmitter(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cuda_lib")
                {
                }
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion