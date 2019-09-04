// Microsoft (c) 2019, NNFusion Team

#include <iostream>
#include <stdio.h>
#include <vector>

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class InvertPermutation : public KernelEmitter
            {
            public:
                InvertPermutation(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cuda")
                {
                    data_size = ctx->inputs[0].get_size();
                }

                LanguageUnit_p emit_function_body() override
                {
                    auto code = ngraph::op::create_code_from_template(
                        R"(
                            for (int i = 0; i < @number@; i++)
                            {
                                output0[input0[i]] = i;
                            }
                        )",
                        {{"number", data_size}});

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu.block_begin();
                    lu << code << "\n";
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    return _lu;
                }

            private:
                int data_size;
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER(
    "InvertPermutation",
    Device(CUDA_GPU).TypeConstraint(DT_FLOAT), // TODO: this op input and output will all be int
    cuda::InvertPermutation)

REGISTER_KERNEL_EMITTER("InvertPermutation",
                        Device(GENERIC_CPU)
                            .TypeConstraint(DT_FLOAT)
                            .Tag("reference"), // TODO: this op input and output will all be int
                        cuda::InvertPermutation)