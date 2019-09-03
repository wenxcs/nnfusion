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
            class BatchGemmFixed : public CudaEmitter
            {
                shared_ptr<ngraph::op::GenericOp> generic_op;

            public:
                BatchGemmFixed(shared_ptr<KernelContext> ctx)
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

                    // Check conditions that pair of inputs must satisfy to run BatchMatMul
                    generic_op->validate_and_infer_types();

                    bool transA = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
                    bool transB = generic_op->localOpConfig.getRoot()["adj_y"]["b"];

                    if (transA || transB)
                        return nullptr;

                    if (ctx->outputs[0].get_element_type().c_type_string() != "float")
                        return nullptr;

                    ngraph::Shape input_shape_0 = generic_op->get_input_shape(0);
                    ngraph::Shape input_shape_1 = generic_op->get_input_shape(1);
                    if (input_shape_0.size() != input_shape_1.size())
                        return nullptr;

                    std::reverse(input_shape_0.begin(), input_shape_0.end());
                    std::reverse(input_shape_1.begin(), input_shape_1.end());
                    size_t batch_0 = 1, batch_1 = 1;
                    for (int i = 2; i < input_shape_0.size(); ++i)
                        batch_0 *= input_shape_0[i], batch_1 *= input_shape_1[i];

                    assert(input_shape_0.size() >= 2 && input_shape_1.size() >= 2);
                    assert(batch_0 == batch_1);
                    input_shape_0.resize(2), input_shape_1.resize(2);
                    input_shape_0.push_back(batch_0), input_shape_1.push_back(batch_1);
                    std::reverse(input_shape_0.begin(), input_shape_0.end());
                    std::reverse(input_shape_1.begin(), input_shape_1.end());

                    assert(input_shape_0[2] == input_shape_1[1]);

                    std::string templ;
                    if (input_shape_0 == ngraph::Shape({16, 512, 512}) &&
                        input_shape_1 == ngraph::Shape({16, 512, 64}))
                    {
                        templ =
                            "rocm_adapter/fixed_kernels/batch_gemm/"
                            "batch_matmul_autotvm_NN_16x512x512x64.h.in";
                        m_gridDim = dim3(1, 16, 16);
                        m_blockDim = dim3(32, 8, 1);
                    }
                    else if (input_shape_0 == ngraph::Shape({16, 512, 64}) &&
                             input_shape_1 == ngraph::Shape({16, 64, 512}))
                    {
                        templ =
                            "rocm_adapter/fixed_kernels/batch_gemm/"
                            "batch_matmul_autotvm_NN_16x512x64x512.h.in";
                        m_gridDim = dim3(1, 16, 16);
                        m_blockDim = dim3(32, 8, 1);
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

REGISTER_KERNEL_EMITTER("BatchMatMul",                                               // op_name
                        Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("PRIORITY_2"), // attrs
                        cuda::BatchGemmFixed)                                        // constructor
