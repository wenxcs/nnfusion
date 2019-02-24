// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for ir::anyop
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/cuda/op/pad.hpp"
#include "../test_util/common.hpp"

const static std::string def_code =
    R"(extern "C" __global__ void cuda_pad_float_float_float2pad_i2_3pad_o7_6_pb1_0_pi2_1(float* in, float* pad, float* out, size_t n)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        size_t input_shape0 = 2;
        size_t input_shape1 = 3;
        uint32_t input_strides0 = 3;
        uint32_t input_strides1 = 1;
        uint32_t output_strides0 = 6;
        uint32_t output_strides1 = 1;
        uint32_t padding_below0 = 1;
        uint32_t padding_below1 = 0;
        uint32_t padding_interior0 = 2;
        uint32_t padding_interior1 = 1;
        bool in_bounds = true;
        uint32_t output_pixel = tid;
        uint32_t input_pixel = 0;
        int32_t input, input_dil;
        input_dil = output_pixel / output_strides0 - padding_below0;
        input = input_dil / (padding_interior0 + 1);
        input_dil %= (padding_interior0 + 1);
        in_bounds = in_bounds && (input >= 0) && (input < input_shape0) && (input_dil == 0);
        input_pixel += input * input_strides0;
        output_pixel %= output_strides0;
        input_dil = output_pixel / output_strides1 - padding_below1;
        input = input_dil / (padding_interior1 + 1);
        input_dil %= (padding_interior1 + 1);
        in_bounds = in_bounds && (input >= 0) && (input < input_shape1) && (input_dil == 0);
        input_pixel += input * input_strides1;
        out[tid] = (in_bounds) ? in[input_pixel] : *pad;
    }
}
)";

TEST(nnfusion_cuda_op, pad)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Pad>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Pad::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Pad>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Pad::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == def_code);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cuda_pad_float_float_float2pad_i2_3pad_o7_6_pb1_0_pi2_1");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cuda_pad_float_float_float2pad_i2_3pad_o7_6_pb1_0_pi2_1_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == def_code);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cuda") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        auto value = nnfusion::inventory::generate_input<op::Pad, float>(0);
        in.push_back(vector<float>(value.begin(), value.begin() + 6));
        in.push_back(vector<float>(value.begin() + 6, value.begin() + 7));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Pad, float>(0));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}

TEST(nnfusion_cuda_op, pad_exterior_2d_0x3)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Pad>(1);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Pad::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Pad>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Pad::codegen(op);

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        auto value = nnfusion::inventory::generate_input<op::Pad, float>(1);
        in.push_back(vector<float>());
        in.push_back(value);
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Pad, float>(1));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}