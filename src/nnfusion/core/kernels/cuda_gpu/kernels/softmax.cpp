// Microsoft (c) 2019, NNFusion Team

#include "softmax.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Softmax::Softmax(shared_ptr<KernelContext> ctx)
    : CudaEmitter(ctx)
{
    auto node = static_pointer_cast<ngraph::op::Softmax>(ctx->node);
    input_shape = ngraph::Shape(ctx->inputs[0].get_shape());
    output_shape = ngraph::Shape(ctx->outputs[0].get_shape());

    // this kernel currently can only handle 2D matrix, thus we have to transfer a >2D tensor
    // to 2D softmax
    axes = node->get_axes();
    std::vector<size_t> axes_flag(input_shape.size(), 0);
    for (auto const& axis : axes)
    {
        axes_flag[axis] = 1;
    }
    height = 1;
    width = 1;
    int i = 0;
    for (; i < axes_flag.size() && axes_flag[i] == 0; i++)
    {
        height *= input_shape[i];
    }
    for (; i < axes_flag.size(); i++)
    {
        if (axes_flag[i] == 0)
        {
            valid_inputs = false;
            break;
        }
        width *= input_shape[i];
    }

    expected_block_size =
        width > 512 ? 512 : pow(2, static_cast<size_t>(log2(static_cast<float>(width))));

    std::stringstream tag;
    tag << join(input_shape, "_") << "_h" << height << "_w" << width;
    custom_tag = tag.str();
}

LanguageUnit_p cuda::Softmax::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)

    // this kernel currently can only handle 2D matrix, where the reduce_axes is 1

    if (valid_inputs)
    {
        auto code = ngraph::op::create_code_from_template(
            R"(
int height = @height@;
int width = @width@;
int block_size = @block_size@;
const int warp_size = @warp_size@;
__shared__ float shm[warp_size];

for (int block_idx = blockIdx.y * blockDim.x + blockIdx.x; block_idx < height;
     block_idx += blockDim.x * blockDim.y) {
    int thread_idx = threadIdx.x;
    int data_idx_offset = block_idx * width;
    
    if (threadIdx.x < block_size) {
        // 1. find the max number, max value is stored in shm[0]
        float val = -1.e20;
        for (int tidx = thread_idx; tidx < width; tidx += block_size) {
            int data_idx = tidx + data_idx_offset;
            if (val < input0[data_idx]) val = input0[data_idx];
        }
        __syncthreads();
        val = reduceMax(val, thread_idx, block_size, shm);
        if (0 == thread_idx) shm[0] = val;
        __syncthreads();

        // 2. sub max Value and do Exp operation
        // value that is larger than kThreshold will be truncated when calculating
        // logitis in softmax.
        const float kThreshold = 64.;
        float sum = 0.;
        for (int tidx = thread_idx; tidx < width; tidx += block_size) {
            int data_idx = tidx + data_idx_offset;
            float sub = input0[data_idx] - shm[0];
            if (sub < -kThreshold) sub = -kThreshold;
            //input0[data_idx] = sub; // TODO: check inplace update
            float exp_val = expf(sub);
            sum += exp_val;
            output0[data_idx] = exp_val;
        }
        __syncthreads();

        // 3. reduce sum
        val = reduceSum(sum, thread_idx, block_size, shm);
        if (thread_idx == 0) shm[0] = val;
        __syncthreads();

        // 4. divideed by sum
        for (int tidx = thread_idx; tidx < width; tidx += block_size) {
            int data_idx = tidx + data_idx_offset;
            output0[data_idx] /= shm[0];
        }
    } 
}
        )",
            {{"height", height},
             {"width", width},
             {"block_size", expected_block_size},
             {"warp_size", 32}});

        lu << code << "\n";
    }
    else
    {
        return nullptr;
    }

    return _lu;
}

void cuda::Softmax::set_launch_config()
{
    m_gridDim = dim3(height, 1, 1);
    m_blockDim = dim3(expected_block_size, 1, 1);
}

LanguageUnit_p cuda::Softmax::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(declaration::cuda_reduce_primitive);
    return _lu;
}

REGISTER_KERNEL_EMITTER("Softmax",                                                    // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::Softmax)                                                // constructor