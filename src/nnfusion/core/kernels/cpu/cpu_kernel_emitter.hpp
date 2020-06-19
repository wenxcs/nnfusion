// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/core/kernels/cpu/cpu_langunit.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

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
                LanguageUnit_p emit_function_signature() override;
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
                    m_intra_op_parallelism = true;
                }

                LanguageUnit_p emit_eigen_utils();

            protected:
                std::string emit_eigen_vector(const shared_ptr<nnfusion::descriptor::Tensor> tw,
                                              const string& name = "");
                std::string emit_eigen_matrix(const shared_ptr<nnfusion::descriptor::Tensor> tw,
                                              const string& name = "");
            };

            class MlasKernelEmitter : public CpuKernelEmitter
            {
            public:
                MlasKernelEmitter(shared_ptr<KernelContext> ctx)
                    : CpuKernelEmitter(ctx)
                {
                    m_intra_op_parallelism = true;
                }
            };

            class AntaresCpuKernelEmitter : public CpuKernelEmitter
            {
            public:
                AntaresCpuKernelEmitter(shared_ptr<KernelContext> ctx)
                    : CpuKernelEmitter(ctx)
                {
                    m_intra_op_parallelism = true;
                    initialize(nnfusion::op::get_translation(ctx->gnode));
                }

                virtual LanguageUnit_p emit_function_body() override;
                virtual LanguageUnit_p emit_dependency() override;

                virtual void initialize(const std::string& expression);

            protected:
                std::string m_expression;
                std::string m_args;
                static std::unordered_map<std::string, std::string> s_cached_kernels;
            };

            class SimdKernelEmitter : public CpuKernelEmitter
            {
            public:
                SimdKernelEmitter(shared_ptr<KernelContext> ctx)
                    : CpuKernelEmitter(ctx)
                {
                    m_intra_op_parallelism = true;
                }

                virtual std::pair<std::string, shared_ptr<LanguageUnit>> get_op_kernel()
                {
                    return std::make_pair("", nullptr);
                }

            protected:
                const uint32_t m_simd_block_size = 8;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
