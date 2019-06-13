// Microsoft (c) 2019, NNFusion Team
#pragma once

// This is the 2rd-generation of kernel definition, recommend to extend new ops with this style
// Changes needed for creating an new kernel with 2rd generation style:
//
// 3 files to change:
//   [a] ./new_kernel_0.cpp
//   [b] ./new_kernel_0.hpp
//   [c] ../../../ops/op_registration.cpp

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

/*********************************
>> For Config detail, please reference ../../../ops/op_registration.cpp

Example:
    BatchMatmul::Config {
        "adj_x": {
            "b": false,
        },
        "adj_y": {
            "b": false,
        },
    }
*********************************/

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class BatchMatmul : public CudaLibEmitter
            {
                shared_ptr<ngraph::op::GenericOp> generic_op;

            public:
                BatchMatmul(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                    , generic_op(static_pointer_cast<ngraph::op::GenericOp>(ctx->node))
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    // function signature:
                    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
                    lu.block_begin();
                    lu << "// hello text for batch_matmul, "
                       << generic_op->localOpConfig.get("adj_x")["b"] << ";\n";
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register Pad kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("BatchMatmul",                                                // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::BatchMatmul)                                            // constructor
