// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "cuda_helper.hpp"

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
                ~CudaEmitter(){};

                // config the blockDim and gridDim
                virtual void set_launch_config() = 0;

                LanguageUnit_p emit_function_call() override;
                
                LanguageUnit_p emit_function_signature() override;

            protected:
                dim3 m_blockDim;
                dim3 m_gridDim;
            };

            class CudnnEmitter : public KernelEmitter
            {
            public:
                CudnnEmitter(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cudnn")
                {
                }
                ~CudnnEmitter(){};

                // A unified way to generate test code for cuda kernels
                // LanguageUnit_p emit_test();
            };

            // class ElementwiseKernel : public CudaEmitter
            // {
            // public:
            //     ElementwiseKernel(shared_ptr<KernelContext> ctx)
            //         : KernelEmitter(ctx),
            //         m_kernel_type("cuda_elementwise")
            //     {
            //     }
            //     ~ElementwiseKernel();

            //     // e.g., tanhf, sigmoidf, +, -, etc.
            //     virtual string operator_func() = 0;

            //     virtual void set_launch_config() override;
            //     virtual LanguageUnit_p emit_function_body() override;
            // };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion