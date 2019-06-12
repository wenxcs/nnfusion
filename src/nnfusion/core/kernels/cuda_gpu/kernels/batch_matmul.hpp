// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "../../../ops/generic_op.hpp"

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
            class BatchMatmul : public CudaEmitter
            {
				shared_ptr<ngraph::op::GenericOp> generic_op;

            public:
                BatchMatmul(shared_ptr<KernelContext> ctx) : CudaEmitter(ctx), generic_op(static_pointer_cast<ngraph::op::GenericOp>(ctx->node)) {
				}

                LanguageUnit_p emit_function_body() override {
					LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
					auto& lu = *_lu;
					// function signature:
					// extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
					lu.block_begin();
					lu << "// hello text for batch_matmul, " << generic_op->localOpConfig.get("adj_x")["b"] << ";\n";
					lu.block_end();
					return _lu;
				}

                void set_launch_config() override {
					// Just for test currently
					m_gridDim = dim3(4, 1, 1);
					m_blockDim = dim3(64, 1, 1);
				}

                LanguageUnit_p emit_dependency() override {
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

