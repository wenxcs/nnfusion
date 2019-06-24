// Microsoft (c) 2019, NNFusion Team
#pragma once

// This is the 2rd-generation of kernel definition, recommend to extend new ops with this style
// Changes needed for creating an new kernel with 2rd generation style.
//

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

/*********************************

REGISTER_OP(OneHot)
    .attr<int>("axis", -1)
    .attr<int>("depth")
    .attr<ngraph::op::OpConfig::any>("off_value", 1.0f)
    .attr<ngraph::op::OpConfig::any>("on_value", 0.0f)
    ...

*********************************/

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class OneHot : public CudaEmitter
            {
                shared_ptr<ngraph::op::GenericOp> generic_op;

            public:
                OneHot(shared_ptr<KernelContext> ctx)
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
                    assert(axis == input_shape_0.size() - 1);
                    size_t groups = 1;
                    for (int i = 0; i < input_shape_0.size(); ++i)
                        groups *= input_shape_0[i];
                    cfg["groups"] = groups;

                    auto code = ngraph::op::create_code_from_template(
                        R"(
    int idx = blockIdx * blockDim.x + threadIdx;
    if (idx >= @groups@)
        return;
    for (int i = 0; i < @depth@; ++i)
        output0[idx * @depth@ + i] = @off_value@;
    output0[idx * @depth@ + input0[idx]] = @on_value@;
)",
                        {
                            {"groups", groups},
                            {"depth", cfg["depth"]},
                            {"off_value", cfg["off_value"]},
                            {"on_value", cfg["on_value"]},
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
                    int groups = generic_op->localOpConfig.getRoot()["groups"];
                    m_gridDim = dim3((groups + 63) / 64, 1, 1);
                    m_blockDim = dim3(64, 1, 1);
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("OneHot",                                                     // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::OneHot)                                                 // constructor
