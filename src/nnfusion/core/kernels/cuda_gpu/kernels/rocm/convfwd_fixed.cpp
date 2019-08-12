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
                    bool using_fixed = getenv("NNFUSION_ENABLE_FIXED")
                                           ? bool(atoi(getenv("NNFUSION_ENABLE_FIXED")))
                                           : 1;
                    if (!using_fixed)
                        return nullptr;

                    GENERIC_OP_LOGGING();
                    auto& ctx = m_context;

                    auto& input_shape = ctx->inputs[0].get_shape();
                    auto& filter_shape = ctx->inputs[1].get_shape();
                    auto& output_shape = ctx->outputs[0].get_shape();

                    auto conv = static_pointer_cast<ngraph::op::Convolution>(ctx->node);
                    auto& window_dilation_strides = conv->get_window_dilation_strides();
                    auto& window_movement_strides = conv->get_window_movement_strides();
                    auto& data_dilation_strides = conv->get_data_dilation_strides();
                    auto& padding_below_diff = conv->get_padding_below();
                    auto& padding_above_diff = conv->get_padding_above();
                    auto& dtype = ctx->outputs[0].get_element_type().c_type_string();

                    if (dtype != "float")
                        return nullptr;

                    // generic_op->validate_and_infer_types();
                    // auto& cfg = generic_op->localOpConfig.getRoot();
                    auto matching = [&](const ngraph::Shape& _input_shape,
                                        const ngraph::Shape& _filter_shape,
                                        const ngraph::Shape& _output_shape,
                                        const ngraph::Strides& _dilation,
                                        const ngraph::Strides& _data_dilation,
                                        const ngraph::Strides& _stride,
                                        const ngraph::CoordinateDiff& _padding_below_diff,
                                        const ngraph::CoordinateDiff& _padding_above_diff) -> bool {
                        if (input_shape != _input_shape)
                            return false;
                        if (filter_shape != _filter_shape)
                            return false;
                        if (output_shape != _output_shape)
                            return false;
                        if (window_dilation_strides != _dilation)
                            return false;
                        if (data_dilation_strides != _data_dilation)
                            return false;
                        if (window_movement_strides != _stride)
                            return false;
                        if (padding_below_diff != _padding_below_diff)
                            return false;
                        if (padding_above_diff != _padding_above_diff)
                            return false;
                        return true;
                    };
                    std::string templ;
                    if (matching({128, 3, 227, 227},
                                 {96, 3, 11, 11},
                                 {128, 96, 55, 55},
                                 {1, 1},
                                 {1, 1},
                                 {4, 4},
                                 {0, 0},
                                 {0, 0}))
                    {
                        templ =
                            "rocm_adapter/fixed_kernels/convfwd/"
                            "conv2d_fwd_128_3_227_227_96_11_11_4_0_1.h.in";
                        m_gridDim = dim3(1, 55, 128);
                        m_blockDim = dim3(5, 1, 48);
                    }
                    else
                        return nullptr;

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu.block_begin();
                    lu << nnfusion::codegen::get_content_from_templates(templ) << "\n";
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

                void set_launch_config() override { GENERIC_OP_LOGGING(); }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Convolution",                                                // op_name
                        Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::ConvFwdFixed)                                           // constructor
