// Microsoft (c) 2019, Yuchao
/**
 * \brief Unit tests for Softmax
 * \author Yuchao Zheng
 */

#include "ngraph/runtime/nnfusion/cuda/op/softmax.hpp"
#include "../test_util/common.hpp"

const static std::string softmax_0_def =
    R"(extern "C" __global__ void cuda_softmax_float_float_i_2_2_3_axis_0(float* in, float* out, size_t nthreads)
{
    uint32_t non_reduce_strides0 = 1;
    uint32_t non_reduce_strides_in_input0 = 1;
    uint32_t reduce_shape0 = 2;
    uint32_t reduce_strides_in_input0 = 6;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nthreads)
    {
        uint32_t dim_idx_generator = tid;
        uint32_t in_idx = 0;
        in_idx += (dim_idx_generator / non_reduce_strides0) * non_reduce_strides_in_input0;
        dim_idx_generator %= non_reduce_strides0;
        uint32_t init_in_idx = in_idx;
        float r_max = in[init_in_idx];
        float input_i;
        {
            uint32_t reduce_idx = in_idx;
            uint32_t step = reduce_strides_in_input0;
            if(reduce_idx != init_in_idx)
            {
                input_i = in[reduce_idx];
                r_max = r_max > input_i ? r_max : input_i;
            }
            reduce_idx += step;
            int idx0 = 1;
            for(; idx0 + 8 - 1 < reduce_shape0; idx0 += 8)
            {
                input_i = in[reduce_idx];
                r_max = r_max > input_i ? r_max : input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                r_max = r_max > input_i ? r_max : input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                r_max = r_max > input_i ? r_max : input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                r_max = r_max > input_i ? r_max : input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                r_max = r_max > input_i ? r_max : input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                r_max = r_max > input_i ? r_max : input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                r_max = r_max > input_i ? r_max : input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                r_max = r_max > input_i ? r_max : input_i;
                reduce_idx += step;
            }
            for(; idx0 < reduce_shape0; idx0++)
            {
                input_i = in[reduce_idx];
                r_max = r_max > input_i ? r_max : input_i;
                reduce_idx += step;
            }
        }
        float r_sum = 0;
        float c = 0;
        float y;
        float t;
        {
            uint32_t reduce_idx = in_idx;
            uint32_t step = reduce_strides_in_input0;
            int idx0 = 0;
            for(; idx0 + 8 - 1 < reduce_shape0; idx0 += 8)
            {
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max);
                y = input_i - c;
                t = r_sum + y;
                c = (t - r_sum) - y;
                r_sum = t;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max);
                y = input_i - c;
                t = r_sum + y;
                c = (t - r_sum) - y;
                r_sum = t;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max);
                y = input_i - c;
                t = r_sum + y;
                c = (t - r_sum) - y;
                r_sum = t;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max);
                y = input_i - c;
                t = r_sum + y;
                c = (t - r_sum) - y;
                r_sum = t;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max);
                y = input_i - c;
                t = r_sum + y;
                c = (t - r_sum) - y;
                r_sum = t;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max);
                y = input_i - c;
                t = r_sum + y;
                c = (t - r_sum) - y;
                r_sum = t;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max);
                y = input_i - c;
                t = r_sum + y;
                c = (t - r_sum) - y;
                r_sum = t;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max);
                y = input_i - c;
                t = r_sum + y;
                c = (t - r_sum) - y;
                r_sum = t;
                reduce_idx += step;
            }
            for(; idx0 < reduce_shape0; idx0++)
            {
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max);
                y = input_i - c;
                t = r_sum + y;
                c = (t - r_sum) - y;
                r_sum = t;
                reduce_idx += step;
            }
        }
        {
            uint32_t reduce_idx = in_idx;
            uint32_t step = reduce_strides_in_input0;
            int idx0 = 0;
            for(; idx0 + 8 - 1 < reduce_shape0; idx0 += 8)
            {
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max) / r_sum;
                out[reduce_idx] = input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max) / r_sum;
                out[reduce_idx] = input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max) / r_sum;
                out[reduce_idx] = input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max) / r_sum;
                out[reduce_idx] = input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max) / r_sum;
                out[reduce_idx] = input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max) / r_sum;
                out[reduce_idx] = input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max) / r_sum;
                out[reduce_idx] = input_i;
                reduce_idx += step;
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max) / r_sum;
                out[reduce_idx] = input_i;
                reduce_idx += step;
            }
            for(; idx0 < reduce_shape0; idx0++)
            {
                input_i = in[reduce_idx];
                input_i = exp(input_i - r_max) / r_sum;
                out[reduce_idx] = input_i;
                reduce_idx += step;
            }
        }
    }
}
)";

const static std::string softmax_1_def =
    R"(extern "C" __global__ void cuda_softmax_float_float_i_2_3_axis_1(float* in, float* out)
{
    uint32_t reduce_count = 3;
    uint32_t non_reduce_strides0 = 1;
    uint32_t non_reduce_strides_in_input0 = 3;
    uint32_t reduce_shape0 = 3;
    uint32_t reduce_strides_in_input0 = 1;
    int reduce_strides_magic0 = 1;
    int reduce_strides_shift0 = 0;
    int non_reduce_strides_magic0 = 1;
    int non_reduce_strides_shift0 = 0;
    extern __shared__ float sdata[];
    uint32_t bid = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t step = blockDim.x;
    int coordinate_product = bid;
    int non_reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, non_reduce_strides_magic0, non_reduce_strides_shift0);
    uint32_t non_reduce_input_index = 0;
    non_reduce_input_index += non_reduce_coordinate0 * non_reduce_strides_in_input0;
    uint32_t input_idx;
    uint32_t reduce_idx = tid;
    float r_max;
    float input_i;
    {
        int coordinate_product = reduce_idx;
        int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
        uint32_t reduce_input_index = 0;
        reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
        input_idx = reduce_input_index + non_reduce_input_index;
        input_i = load(in, input_idx);
        r_max = input_i;
        reduce_idx += step;
    }
    while (reduce_idx + 7 * step < reduce_count)
    {
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            r_max = r_max > input_i ? r_max : input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            r_max = r_max > input_i ? r_max : input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            r_max = r_max > input_i ? r_max : input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            r_max = r_max > input_i ? r_max : input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            r_max = r_max > input_i ? r_max : input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            r_max = r_max > input_i ? r_max : input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            r_max = r_max > input_i ? r_max : input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            r_max = r_max > input_i ? r_max : input_i;
            reduce_idx += step;
        }
    }
    while (reduce_idx < reduce_count)
    {
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            r_max = r_max > input_i ? r_max : input_i;
            reduce_idx += step;
        }
    }
    input_i = __shfl_down_sync(0xffffffff, r_max, 1, 32);
    r_max = r_max > input_i ? r_max : input_i;
    if(tid == 0)
    {
        sdata[0] = r_max;
    }
    __syncthreads();
    r_max = sdata[0];
    float r_sum = 0;
    float c = 0;
    float y;
    float t;
    reduce_idx = tid;
    while (reduce_idx + 7 * step < reduce_count)
    {
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max);
            y = input_i - c;
            t = r_sum + y;
            c = (t - r_sum) - y;
            r_sum = t;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max);
            y = input_i - c;
            t = r_sum + y;
            c = (t - r_sum) - y;
            r_sum = t;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max);
            y = input_i - c;
            t = r_sum + y;
            c = (t - r_sum) - y;
            r_sum = t;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max);
            y = input_i - c;
            t = r_sum + y;
            c = (t - r_sum) - y;
            r_sum = t;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max);
            y = input_i - c;
            t = r_sum + y;
            c = (t - r_sum) - y;
            r_sum = t;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max);
            y = input_i - c;
            t = r_sum + y;
            c = (t - r_sum) - y;
            r_sum = t;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max);
            y = input_i - c;
            t = r_sum + y;
            c = (t - r_sum) - y;
            r_sum = t;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max);
            y = input_i - c;
            t = r_sum + y;
            c = (t - r_sum) - y;
            r_sum = t;
            reduce_idx += step;
        }
    }
    while (reduce_idx < reduce_count)
    {
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max);
            y = input_i - c;
            t = r_sum + y;
            c = (t - r_sum) - y;
            r_sum = t;
            reduce_idx += step;
        }
    }
    r_sum += __shfl_down_sync(0xffffffff, r_sum, 1, 32);
    __syncthreads();
    if(tid == 0)
    {
        sdata[0] = r_sum;
    }
    __syncthreads();
    r_sum = sdata[0];
    reduce_idx = tid;
    while (reduce_idx + 7 * step < reduce_count)
    {
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max) / r_sum;
            out[input_idx] = input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max) / r_sum;
            out[input_idx] = input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max) / r_sum;
            out[input_idx] = input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max) / r_sum;
            out[input_idx] = input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max) / r_sum;
            out[input_idx] = input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max) / r_sum;
            out[input_idx] = input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max) / r_sum;
            out[input_idx] = input_i;
            reduce_idx += step;
        }
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max) / r_sum;
            out[input_idx] = input_i;
            reduce_idx += step;
        }
    }
    while (reduce_idx < reduce_count)
    {
        {
            int coordinate_product = reduce_idx;
            int reduce_coordinate0 = division_by_invariant_multiplication(coordinate_product, reduce_strides_magic0, reduce_strides_shift0);
            uint32_t reduce_input_index = 0;
            reduce_input_index += reduce_coordinate0 * reduce_strides_in_input0;
            input_idx = reduce_input_index + non_reduce_input_index;
            input_i = load(in, input_idx);
            input_i = exp(input_i - r_max) / r_sum;
            out[input_idx] = input_i;
            reduce_idx += step;
        }
    }
}
)";

TEST(nnfusion_cuda_op, softmax_axis_3d)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Softmax>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Softmax::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Softmax>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Softmax::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == softmax_0_def);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cuda_softmax_float_float_i_2_2_3_axis_0");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cuda_softmax_float_float_i_2_2_3_axis_0_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == softmax_0_def);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cuda") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        /*intput*/ in.push_back(nnfusion::inventory::generate_input<op::Softmax, float>(0));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Softmax, float>(0));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}

TEST(nnfusion_cuda_op, softmax_axis)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Softmax>(1);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Softmax::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Softmax>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Softmax::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == softmax_1_def);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cuda_softmax_float_float_i_2_3_axis_1");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cuda_softmax_float_float_i_2_3_axis_1_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == softmax_1_def);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cuda") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        /*intput*/ in.push_back(nnfusion::inventory::generate_input<op::Softmax, float>(1));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Softmax, float>(1));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}