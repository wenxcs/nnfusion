// Microsoft (c) 2019, Yuchao
/**
 * \brief Unit tests for result
 * \author Yuchao Zheng
 */

#include "ngraph/runtime/nnfusion/cuda/op/result.hpp"
#include "../test_util/common.hpp"

const static std::string result_0_def =
    R"(// No codegen for Result since it's memcpy().
)";

TEST(nnfusion_cuda_op, result)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Result>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Result::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Result>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Result::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == result_0_def);
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
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == result_0_def);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cuda") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        in.push_back(nnfusion::inventory::generate_input<op::Result, float>(0));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Result, float>(0));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}
