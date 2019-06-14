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
    BatchMatMul::Config {
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
            class BatchMatMul : public CudaLibEmitter
            {
                shared_ptr<ngraph::op::GenericOp> generic_op;

            public:
                BatchMatMul(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                    , generic_op(static_pointer_cast<ngraph::op::GenericOp>(ctx->node))
                {
					GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_body() override
                {
					GENERIC_OP_LOGGING();

					const ngraph::Shape& input_shape_0 = generic_op->get_input_shape(0);
					const ngraph::Shape& input_shape_1 = generic_op->get_input_shape(1);

					// Check conditions that pair of inputs must satisfy to run BatchMatMul
					generic_op->validate_and_infer_types();

					// Handle matmul without transpose
					assert(generic_op->localOpConfig.getRoot()["adj_x"]["b"] == false);
					assert(generic_op->localOpConfig.getRoot()["adj_y"]["b"] == false);

					size_t A1 = 1LU; for (int i = input_shape_0.size() - 3; i >= 0; --i) A1 *= input_shape_0[i];
					int A2 = input_shape_0[input_shape_0.size() - 2];
					int A3 = input_shape_0[input_shape_0.size() - 1];
					int A4 = input_shape_1[input_shape_0.size() - 1];

					int m = A4, n = A2, k = A3, lda = A4, stride_a = A3 * A4, ldb = A3, stride_b = A2 * A3, ldc = A4, stride_c = A2 * A4;

					auto code = ngraph::op::create_code_from_template(R"(
						static const float alpha = 1.0f, beta = 0.0f;
						if (!@hCublas@)
							assert(CUBLAS_STATUS_SUCCESS == @api_create@(&@hCublas@));
						assert(CUBLAS_STATUS_SUCCESS == @api_exec@(
							global_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, @m@, @n@, @k@,
							&alpha, input0, @lda@, @stride_a@, input1, @ldb@, @stride_b@,
							&beta, output0, @ldc@, @stride_c@, @batch@));
					)", {
						{"hCublas", "global_cublas_handle"},
						{"api_create", "cublasCreate"},
						{"api_exec", "cublasSgemmStridedBatched"},
						{"m", m},
						{"n", n},
						{"k", k},
						{"lda", lda},
						{"ldb", ldb},
						{"ldc", ldc},
						{"stride_a", stride_a},
						{"stride_b", stride_b},
						{"stride_c", stride_c},
						{"batch", A1},
					});

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    // function signature:
                    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
                    lu.block_begin();
                    lu << code << "\n";
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
					GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu_header(new LanguageUnit(get_function_name() + "_dep"));
                    _lu_header->require(header::cuda);
                    _lu_header->require(header::cublas);
                    _lu_header->require(declaration::global_cublas_handle);
                    _lu_header->require(macro::CUBLAS_SAFE_CALL);
                    return _lu_header;
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register Pad kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("BatchMatMul",                                                // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::BatchMatMul)                                            // constructor
