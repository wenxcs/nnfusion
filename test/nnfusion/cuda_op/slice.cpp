// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit test
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/cuda/op/slice.hpp"
#include "../test_util/common.hpp"

const static std::string source =
    R"(extern "C" __global__ void cuda_slice_float_float_r_2_i_4_4_o_3_2_lb_0_1_ss_1_1(float* in, float* out, uint32_t n)
{
    uint32_t input_strides[] = {4, 1};
    uint32_t output_strides[] = {2, 1};
    uint32_t lower_bounds[] = {0, 1};
    uint32_t slice_strides[] = {1, 1};
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        uint32_t input_idx = 0;
        uint32_t output_idx = tid;
        input_idx += (((output_idx / output_strides[0]) * slice_strides[0]) + lower_bounds[0]) * input_strides[0];
        output_idx %= output_strides[0];
        input_idx += (((output_idx / output_strides[1]) * slice_strides[1]) + lower_bounds[1]) * input_strides[1];
        out[tid] = in[input_idx];
    }
}
)";

TEST(nnfusion_cuda_op, slice)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Slice>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Slice::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Slice>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Slice::codegen(op);

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
        EXPECT_TRUE(name == "cuda_slice_float_float_r_2_i_4_4_o_3_2_lb_0_1_ss_1_1");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cuda_slice_float_float_r_2_i_4_4_o_3_2_lb_0_1_ss_1_1_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == source);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        auto inval = nnfusion::inventory::generate_input<op::Slice, float>(0);
        in.push_back(inval);

        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Slice, float>(0));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}