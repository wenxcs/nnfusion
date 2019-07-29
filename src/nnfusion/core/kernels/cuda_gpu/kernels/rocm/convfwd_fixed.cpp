// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class ConvFwdFixed : public CudaEmitter
            {
                shared_ptr<ngraph::op::GenericOp> generic_op;

            public:
                ConvFwdFixed(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                    , generic_op(static_pointer_cast<ngraph::op::GenericOp>(ctx->node))
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();
                    auto& ctx = m_context;

                    auto& input_shape_0 = ctx->inputs[0].get_shape();
                    auto& filter_shape = ctx->inputs[1].get_shape();
                    auto& output_shape = ctx->outputs[0].get_shape();

                    auto conv = static_pointer_cast<ngraph::op::Convolution>(ctx->node);
                    auto& window_dilation_strides = conv->get_window_dilation_strides();
                    auto& window_movement_strides = conv->get_window_movement_strides();
                    auto& data_dilation_strides = conv->get_data_dilation_strides();
                    auto& padding_below_diff = conv->get_padding_below();
                    auto& padding_above_diff = conv->get_padding_above();
                    auto& dtype = ctx->outputs[0].get_element_type().c_type_string();

                    // generic_op->validate_and_infer_types();
                    // auto& cfg = generic_op->localOpConfig.getRoot();

                    if (input_shape_0 != ngraph::Shape({128, 3, 227, 227}))
                        return nullptr;
                    if (filter_shape != ngraph::Shape({96, 3, 11, 11}))
                        return nullptr;
                    if (output_shape != ngraph::Shape({128, 96, 55, 55}))
                        return nullptr;
                    if (window_dilation_strides != ngraph::Strides({1, 1}))
                        return nullptr;
                    if (data_dilation_strides != ngraph::Strides({1, 1}))
                        return nullptr;
                    if (window_movement_strides != ngraph::Strides({4, 4}))
                        return nullptr;
                    if (padding_below_diff != ngraph::CoordinateDiff({0, 0}))
                        return nullptr;
                    if (padding_above_diff != ngraph::CoordinateDiff({0, 0}))
                        return nullptr;
                    if (dtype != "float")
                        return nullptr;

                    auto code = nnfusion::codegen::get_content_from_templates(
                        "rocm_adapter/fixed_kernels/convfwd/"
                        "conv2d_fwd_128_3_227_227_96_11_11_4_0_1.h.in");

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
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
                    GENERIC_OP_LOGGING();

                    // just for conv2d_fwd_128_3_227_227_96_11_11_4_0_1
                    m_gridDim = dim3(1, 55, 128);
                    m_blockDim = dim3(5, 1, 48);
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Convolution",                                                // op_name
                        Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::ConvFwdFixed)                                           // constructor
