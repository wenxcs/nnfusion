// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit test
 * \author wenxh
 */

#include "../test_util/common.hpp"
#include "ngraph/runtime/nnfusion/cuda/op/reduce.hpp"

const static std::string source =
    R"(extern "C" __global__ void cuda_reduce_add_float_float_s_2_2_axis_0_1_scalar(float* in, float* out, size_t nthreads)
{
    extern __shared__ float sdata[];
    uint32_t tid = threadIdx.x;
    uint32_t step = blockDim.x;
    sdata[tid] = 0;
    uint32_t in_idx = tid;
    float r = 0;
    if(in_idx < nthreads)
    {
        r = in[in_idx];
        in_idx += step;
    }
    while(in_idx + (step * 7) < nthreads)
    {
        r = add(r , in[in_idx]);
        in_idx += step;
        r = add(r , in[in_idx]);
        in_idx += step;
        r = add(r , in[in_idx]);
        in_idx += step;
        r = add(r , in[in_idx]);
        in_idx += step;
        r = add(r , in[in_idx]);
        in_idx += step;
        r = add(r , in[in_idx]);
        in_idx += step;
        r = add(r , in[in_idx]);
        in_idx += step;
        r = add(r , in[in_idx]);
        in_idx += step;
    }
    while(in_idx < nthreads)
    {
        r = add(r , in[in_idx]);
        in_idx += step;
    }
    r = add(r, __shfl_down_sync(0xffffffff, r, 2, 32));
    r = add(r, __shfl_down_sync(0xffffffff, r, 1, 32));
    if(tid == 0)
    {
        out[0] = r;
    }
}
)";

TEST(nnfusion_cuda_op, sum_scalar)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Sum>(1);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Sum::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Sum>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Reduce<ngraph::op::Add>::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(nnfusion::library::trim(def->get_code()) == nnfusion::library::trim(source));
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call != nullptr);
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cuda_reduce_add_float_float_s_2_2_axis_0_1");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cuda_reduce_add_float_float_s_2_2_axis_0_1_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        auto inval = nnfusion::inventory::generate_input<op::Sum, float>(1);
        in.push_back(inval);

        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Sum, float>(1));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}

TEST(nnfusion_cuda_op, sum_nd)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Sum>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Sum::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Sum>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Reduce<ngraph::op::Add>::codegen(op);

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        auto inval = nnfusion::inventory::generate_input<op::Sum, float>(0);
        in.push_back(inval);

        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Sum, float>(0));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}