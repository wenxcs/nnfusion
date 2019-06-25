// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "nnfusion/core/kernels/cpu/cpu_langunit.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class CpuKernelEmitter : public KernelEmitter
            {
            public:
                CpuKernelEmitter(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cpu")
                {
                }
            };

            class MklKernelEmitter : public CpuKernelEmitter
            {
            public:
                MklKernelEmitter(shared_ptr<KernelContext> ctx)
                    : CpuKernelEmitter(ctx)
                {
                }
            };

            class EigenKernelEmitter : public CpuKernelEmitter
            {
            public:
                EigenKernelEmitter(shared_ptr<KernelContext> ctx)
                    : CpuKernelEmitter(ctx)
                {
                }

                LanguageUnit_p emit_eigen_utils();

            protected:
                std::string emit_eigen_vector(const TensorWrapper& tw, const string& name = "");
                std::string emit_eigen_matrix(const TensorWrapper& tw, const string& name = "");
            };

        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion