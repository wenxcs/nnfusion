// Microsoft (c) 2019, Yuchao
/**
 * \brief Unit tests for Broadcast
 * \author Yuchao Zheng
 */

#include "ngraph/runtime/nnfusion/cuda/op/broadcast.hpp"
#include "../test_util/common.hpp"
#include "ngraph/runtime/nnfusion/cuda/op/result.hpp"

const static std::string broadcast_0_def =
    R"(extern "C" __global__ void cuda_broadcast_float_float_r1_s4_rs0(float* in, float* out, size_t nthreads)
{
    uint32_t strides0 = 1;
    int stride_magic0 = 1;
    int stride_shift0 = 0;
    uint32_t reduced_strides0 = 0;
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < nthreads)
    {
        int coordinate_product = tid;
        int coordinate0 = division_by_invariant_multiplication(coordinate_product, stride_magic0, stride_shift0);
        int reduced_idx = 0;
        reduced_idx += coordinate0 * reduced_strides0;
        out[tid] = load(in, reduced_idx);
    }
}
)";

const static std::string broadcast_1_def =
    R"(// No codegen for Result since it's memcpy().
)";

TEST(nnfusion_cuda_op, broadcast)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Broadcast>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Broadcast::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Broadcast>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Broadcast::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == broadcast_0_def);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cuda_broadcast_float_float_r1_s4_rs0");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cuda_broadcast_float_float_r1_s4_rs0_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == broadcast_0_def);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cuda") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        in.push_back(nnfusion::inventory::generate_input<op::Broadcast, float>(0));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Broadcast, float>(0));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}

TEST(nnfusion_cuda_op, broadcast_same_shape)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Broadcast>(1);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Broadcast::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Result>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Result::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == broadcast_1_def);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cuda_result");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cuda_result_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == broadcast_1_def);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cuda") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        in.push_back(nnfusion::inventory::generate_input<op::Broadcast, float>(1));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Broadcast, float>(1));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}
