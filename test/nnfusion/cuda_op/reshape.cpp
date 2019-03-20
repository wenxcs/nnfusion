// Microsoft (c) 2019, Yuchao
/**
 * \brief Unit tests for Reshape
 * \author Yuchao Zheng
 */

#include "ngraph/runtime/nnfusion/cuda/op/reshape.hpp"
#include "../test_util/common.hpp"

const static std::string reshape_2_def =
    R"(extern "C" __global__ void cuda_reshape_float_float_r_6_i_2_2_3_3_2_4_o_2_4_0_5_3_1(float* in, float* out)
{
    uint32_t input_strides0 = 144;
    uint32_t input_strides1 = 72;
    uint32_t input_strides2 = 24;
    uint32_t input_strides3 = 8;
    uint32_t input_strides4 = 4;
    uint32_t input_strides5 = 1;
    uint32_t trans_strides0 = 24;
    uint32_t trans_strides1 = 1;
    uint32_t trans_strides2 = 96;
    uint32_t trans_strides3 = 2;
    uint32_t trans_strides4 = 48;
    uint32_t trans_strides5 = 6;
    size_t n = 288;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        uint32_t input_idx = tid;
        uint32_t output_idx = 0;
        output_idx += (input_idx / input_strides0) * trans_strides0;
        input_idx %= input_strides0;
        output_idx += (input_idx / input_strides1) * trans_strides1;
        input_idx %= input_strides1;
        output_idx += (input_idx / input_strides2) * trans_strides2;
        input_idx %= input_strides2;
        output_idx += (input_idx / input_strides3) * trans_strides3;
        input_idx %= input_strides3;
        output_idx += (input_idx / input_strides4) * trans_strides4;
        input_idx %= input_strides4;
        output_idx += (input_idx / input_strides5) * trans_strides5;
        out[output_idx] = in[tid];
    }
}
)";

const static std::string reshape_1_def =
    R"(extern "C" __global__ void cuda_reshape_float_float_r_0_2_1_i_2_3_4(float* in, float* out)
{
    uint32_t input_strides0 = 12;
    uint32_t input_strides1 = 4;
    uint32_t input_strides2 = 1;
    uint32_t trans_strides0 = 12;
    uint32_t trans_strides1 = 1;
    uint32_t trans_strides2 = 3;
    size_t nx = 4;
    size_t ny = 3;
    size_t nz = 2;
    __shared__ float tile[1][16][17];
    uint32_t base2 = blockIdx.x * blockDim.x;
    uint32_t base1 = blockIdx.y * blockDim.y;
    uint32_t base0 = blockIdx.z * blockDim.z;
    uint32_t tid2 = threadIdx.x;
    uint32_t tid1 = threadIdx.y;
    uint32_t tid0 = threadIdx.z;
    uint32_t otid2 = tid2;
    uint32_t otid1 = tid1;
    uint32_t otid0 = tid0;
    uint32_t idx2 = base2 + tid2;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx2 < nx && idx1 < ny && idx0 < nz)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        input_idx += input_strides2* idx2;
        tile[tid0][tid1][tid2] = in[input_idx];
    }
    otid2 = tid1;
    otid1 = tid2;
    idx2 = base2 + otid2;
    idx1 = base1 + otid1;
    idx0 = base0 + otid0;
    __syncthreads();
    if (idx2 < nx && idx1 < ny && idx0 < nz)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        output_idx += trans_strides2* idx2;
        out[output_idx] = tile[otid0][otid1][otid2];
    }
}
)";

const static std::string reshape_0_def =
    R"(extern "C" __global__ void cuda_reshape_float_float_i_3_3_o_1_0(float* in, float* out)
{
    uint32_t input_strides0 = 3;
    uint32_t input_strides1 = 1;
    uint32_t trans_strides0 = 1;
    uint32_t trans_strides1 = 3;
    size_t nx = 3;
    size_t ny = 3;
    __shared__ float tile[16][17];
    uint32_t base1 = blockIdx.x * blockDim.x;
    uint32_t base0 = blockIdx.y * blockDim.y;
    uint32_t tid1 = threadIdx.x;
    uint32_t tid0 = threadIdx.y;
    uint32_t idx1 = base1 + tid1;
    uint32_t idx0 = base0 + tid0;
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t input_idx = 0;
        input_idx += input_strides0* idx0;
        input_idx += input_strides1* idx1;
        tile[tid0][tid1] = in[input_idx];
    }
    idx1 = base1 + tid0;
    idx0 = base0 + tid1;
    __syncthreads();
    if (idx1 < nx && idx0 < ny)
    {
        uint32_t output_idx = 0;
        output_idx += trans_strides0* idx0;
        output_idx += trans_strides1* idx1;
        out[output_idx] = tile[tid1][tid0];
    }
}
)";

TEST(nnfusion_cuda_op, reshape_2D)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Reshape>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Reshape::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Reshape>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Reshape::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == reshape_0_def);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cuda_reshape_float_float_i_3_3_o_1_0");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cuda_reshape_float_float_i_3_3_o_1_0_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        // EXPECT_TRUE(cuda_op->definition_unit->get_code() == reshape_0_def);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cuda") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        /*intput*/ in.push_back(nnfusion::inventory::generate_input<op::Reshape, float>(0));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Reshape, float>(0));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}

TEST(nnfusion_cuda_op, reshape_3D)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Reshape>(1);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Reshape::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Reshape>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Reshape::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == reshape_1_def);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cuda_reshape_float_float_r_0_2_1_i_2_3_4");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cuda_reshape_float_float_r_0_2_1_i_2_3_4_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == reshape_1_def);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cuda") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        /*intput*/ in.push_back(nnfusion::inventory::generate_input<op::Reshape, float>(1));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Reshape, float>(1));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}

TEST(nnfusion_cuda_op, reshape_D)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Reshape>(2);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Reshape::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Reshape>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Reshape::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == reshape_2_def);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cuda_reshape_float_float_r_6_i_2_2_3_3_2_4_o_2_4_0_5_3_1");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cuda_reshape_float_float_r_6_i_2_2_3_3_2_4_o_2_4_0_5_3_1_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == reshape_2_def);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cuda") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        /*intput*/ in.push_back(nnfusion::inventory::generate_input<op::Reshape, float>(2));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Reshape, float>(2));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}