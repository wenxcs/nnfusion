// Microsoft (c) 2019, NNFusion Team

#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class GemmFixed : public CudaEmitter
            {
                shared_ptr<ngraph::op::GenericOp> generic_op;

            public:
                GemmFixed(shared_ptr<KernelContext> ctx)
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

                    auto& arg0_shape = ctx->inputs[0].get_shape();
                    auto& arg1_shape = ctx->inputs[1].get_shape();
                    auto& out_shape = ctx->outputs[0].get_shape();

                    auto gemm = static_pointer_cast<ngraph::op::Dot>(ctx->node);
                    auto reduction_axes = gemm->get_reduction_axes_count();
                    auto& dtype = ctx->outputs[0].get_element_type().c_type_string();

                    if (arg0_shape.empty() || arg1_shape.empty())
                        return nullptr;
                    if ((arg0_shape.size() == arg1_shape.size()) &&
                        (arg0_shape.size() == reduction_axes))
                        return nullptr;
                    if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1) &&
                        (reduction_axes == 1))
                        return nullptr;
                    if (dtype != "float")
                        return nullptr;

                    std::string templ;
                    if (arg0_shape == ngraph::Shape({128, 9216}) &&
                        arg1_shape == ngraph::Shape({9216, 4096}))
                    {
                        templ =
                            "rocm_adapter/fixed_kernels/gemm/matmul_autotvm_NN_128x9216x4096.h.in";
                        m_gridDim = dim3(128, 4, 1);
                        m_blockDim = dim3(16, 16, 1);
                    }
                    else if (arg0_shape == ngraph::Shape({128, 4096}) &&
                             arg1_shape == ngraph::Shape({4096, 4096}))
                    {
                        templ =
                            "rocm_adapter/fixed_kernels/gemm/matmul_autotvm_NN_128x4096x4096.h.in";
                        m_gridDim = dim3(128, 4, 1);
                        m_blockDim = dim3(16, 16, 1);
                    }
                    else if (arg0_shape == ngraph::Shape({1, 256}) &&
                             arg1_shape == ngraph::Shape({256, 256}))
                    {
                        templ = "rocm_adapter/fixed_kernels/gemm/manual_NN_1x256x256.h.in";
                        m_gridDim = dim3(1, 256, 1);
                        m_blockDim = dim3(64, 1, 1);
                    }
                    else if (arg0_shape == ngraph::Shape({64, 25088}) &&
                             arg1_shape == ngraph::Shape({25088, 4096}))
                    {
                        templ =
                            "rocm_adapter/fixed_kernels/gemm/matmul_autotvm_NN_64x25088x4096.h.in";
                        m_gridDim = dim3(128, 2, 1);
                        m_blockDim = dim3(16, 16, 1);
                    }
                    else if (arg0_shape == ngraph::Shape({512, 4096}) &&
                             arg1_shape == ngraph::Shape({4096, 1024}))
                    {
                        templ =
                            "rocm_adapter/fixed_kernels/gemm/matmul_autotvm_NN_512x4096x1024.h.in";
                        m_gridDim = dim3(16, 8, 1);
                        m_blockDim = dim3(16, 16, 1);
                    }
                    else if (arg0_shape == ngraph::Shape({512, 1024}) &&
                             arg1_shape == ngraph::Shape({1024, 4096}))
                    {
                        templ =
                            "rocm_adapter/fixed_kernels/gemm/matmul_autotvm_NN_512x1024x4096.h.in";
                        m_gridDim = dim3(64, 8, 1);
                        m_blockDim = dim3(16, 16, 1);
                    }
                    else
                        return nullptr;

                    // generic_op->validate_and_infer_types();
                    // auto& cfg = generic_op->localOpConfig.getRoot();

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

REGISTER_KERNEL_EMITTER("Dot",                                                        // op_name
                        Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::GemmFixed)                                              // constructor
