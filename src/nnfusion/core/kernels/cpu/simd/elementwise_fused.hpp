// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "../cpu_kernel_emitter.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class ElementwiseFused : public SimdKernelEmitter
            {
            public:
                ElementwiseFused(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_name() override;
                LanguageUnit_p emit_comments() override;
                static int unique_func_id;

            private:
                std::shared_ptr<KernelContext> FuseContext();
                void FuseFunctionBody(LanguageUnit& lu);
                std::unordered_map<std::string, std::string> in_args, out_args, local_tensors;
            };

        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
