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

                virtual bool is_static_function() override { return false; }
                // Need to regenerate function call with new assigned launch config(stream).
                LanguageUnit_p emit_function_call() override;

            protected:
                // config the blockDim and gridDim
                virtual void set_launch_config() = 0;

                LanguageUnit_p emit_function_signature() override;

                dim3 m_blockDim;
                dim3 m_gridDim;
            };

            class CudaElementwiseEmitter : public CudaEmitter
            {
            public:
                CudaElementwiseEmitter(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                {
                }

                virtual std::pair<std::string, shared_ptr<LanguageUnit>> get_op_kernel() = 0;
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