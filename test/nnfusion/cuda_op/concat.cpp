// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit test
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/cuda/op/concat.hpp"
#include "../test_util/common.hpp"

const static std::string source =
    R"(extern "C" __global__ void cuda_concat_float_r_3_o_2_8_a_1_i_2_2_i_2_3_i_2_3_kernel_1(float* in0, float* in1, float* in2, float* out, uint32_t output_stride, uint32_t split_output_stride, uint32_t split_input_stride_offset, uint32_t input_offset, uint32_t n)
{
    uint32_t inputs_strides[] = {2, 3, 3};
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
    {
        uint32_t block_id = tid / split_output_stride;
        uint32_t block_idx = tid % split_output_stride;
        uint32_t output_idx = block_id * output_stride + block_idx + split_input_stride_offset;
        out[output_idx] = 1;
        if(block_idx < inputs_strides[0 + input_offset])
        {
            out[output_idx] = in0[block_id * inputs_strides[0 + input_offset] + block_idx];
            return;
        }
        block_idx -= inputs_strides[0 + input_offset];
        if(block_idx < inputs_strides[1 + input_offset])
        {
            out[output_idx] = in1[block_id * inputs_strides[1 + input_offset] + block_idx];
            return;
        }
        block_idx -= inputs_strides[1 + input_offset];
        if(block_idx < inputs_strides[2 + input_offset])
        {
            out[output_idx] = in2[block_id * inputs_strides[2 + input_offset] + block_idx];
            return;
        }
        block_idx -= inputs_strides[2 + input_offset];
    }
}
)";

TEST(nnfusion_cuda_op, concat)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Concat>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Concat::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Concat>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Concat::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == source);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call != nullptr);
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cuda_concat_float_r_3_o_2_8_a_1_i_2_2_i_2_3_i_2_3");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cuda_concat_float_r_3_o_2_8_a_1_i_2_2_i_2_3_i_2_3_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == source);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        auto inval = nnfusion::inventory::generate_input<op::Concat, float>(0);
        in.push_back(vector<float>(inval.begin(), inval.begin() + 4));
        in.push_back(vector<float>(inval.begin() + 4, inval.begin() + 10));
        in.push_back(vector<float>(inval.begin() + 10, inval.begin() + 16));

        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Concat, float>(0));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}