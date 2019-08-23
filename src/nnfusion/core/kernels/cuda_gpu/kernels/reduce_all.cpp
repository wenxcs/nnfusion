// Microsoft (c) 2019, NNFusion Team
#pragma once

// This is the 2rd-generation of kernel definition, recommend to extend new ops with this style
// Changes needed for creating an new kernel with 2rd generation style.
//

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

/*********************************

REGISTER_OP(All)
    .attr<int>("axis", -1)
    .attr<bool>("keep_dims", false)
    ...

*********************************/

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class All : public CudaEmitter
            {
                shared_ptr<ngraph::op::GenericOp> generic_op;

            public:
                All(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                    , generic_op(static_pointer_cast<ngraph::op::GenericOp>(ctx->node))
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();

                    const ngraph::Shape& input_shape_0 = generic_op->get_input_shape(0);

                    generic_op->validate_and_infer_types();
                    auto& cfg = generic_op->localOpConfig.getRoot();

                    int axis = cfg["axis"];
                    assert(axis == -1);
                    assert(std::string("bool") == m_context->dtypes[0]);
                    // assert(axis >= 0 && axis < input_shape_0.size());
                    size_t size = 1;
                    for (int i = 0; i < input_shape_0.size(); ++i)
                        size *= input_shape_0[i];

                    auto code = ngraph::op::create_code_from_template(
                        R"(
        int offset = threadIdx.x;
		extern __shared__ bool cache[1024];
		cache[threadIdx.x] = false;
        while (offset < @size@) {
			cache[threadIdx.x] = cache[threadIdx.x] && input0[offset];
            offset += blockDim.x;
        }
		__syncthreads();

		for (unsigned int s = 1; s < blockDim.x; s *= 2) {
			if (threadIdx.x % (s * 2) == 0)
				cache[threadIdx.x] += cache[threadIdx.x + s];
			__syncthreads();
		}
		output0[0] = cache[0];
)",
                        {
                            {"size", cfg["size"]},
                        });

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    // function signature:
                    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0)
                    lu.block_begin();
                    lu << code << "\n";
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override
                {
                    m_gridDim = dim3(1, 1, 1);
                    m_blockDim = dim3(1024, 1, 1);
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("All",                                                        // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::All)                                                    // constructor