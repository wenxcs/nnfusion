// Microsoft (c) 2019, NNFusion Team

#include "equal.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Equal::Equal(shared_ptr<KernelContext> ctx)
    : CudaEmitter(ctx)
{
    auto equal = static_pointer_cast<ngraph::op::GenericOp>(ctx->node);
    input_shape_0 = ngraph::Shape(ctx->inputs[0].get_shape());
    input_shape_1 = ngraph::Shape(ctx->inputs[1].get_shape());
    output_shape = ngraph::Shape(ctx->outputs[0].get_shape());

    int input0_size = input_shape_0.back();
    int input1_size = input_shape_1.back();
    assert(input0_size == input1_size);

    equal_size = input0_size;

    std::stringstream tag;
    tag << "Equal" << join(input_shape_0, "_") << join(input_shape_1, "_")
        << join(output_shape, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::Equal::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    lu << m_context->dtypes[0] << "* equal0 = input0;\n";
    lu << m_context->dtypes[1] << "* equal1 = input1;\n";
    lu << m_context->dtypes[2] << "* out = output0;\n";

    uint32_t nthreads = equal_size;
    lu << "uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n";
    lu << "if (i < " << nthreads << ")\n";
    lu.block_begin();
    {
        lu << "if( (euqal0[i] - euqal1[i]) < 1e-5f )\n";
        lu.block_begin();
        {
            lu << "out[i] = true;\n";
        }
        lu.block_end();
        lu << "else out[i] = false;\n";
    }
    lu.block_end();

    return _lu;
}

void cuda::Equal::set_launch_config()
{
    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
    m_blockDim = dim3(block_size_x, 1, 1);
}

LanguageUnit_p cuda::Equal::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}
REGISTER_KERNEL_EMITTER("Equal",                                                      // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::Equal)                                                  // constructor