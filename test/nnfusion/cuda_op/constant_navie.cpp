// Microsoft (c) 2019, Yuchao
/**
 * \brief Unit tests for Constant navie
 * \author Yuchao Zheng
 */

#include "../test_util/common.hpp"
#include "ngraph/runtime/nnfusion/cuda/op/constant_naive.hpp"

const static std::string constane_naive_0_def =
    R"(extern "C" void read_const_Constant_0_0(float** out)
{
    std::ifstream bin_file("Constant_0_0.bin" , std::ios::in | std::ios::binary);
    cudaMalloc((void**)out, 32);
    char* tmp_mem = new char[32];
    bin_file.read(tmp_mem, 32);
    cudaMemcpy(*out, tmp_mem, 32, cudaMemcpyHostToDevice);
    bin_file.close();
}
)";

TEST(nnfusion_cuda_op, constane_naive)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Constant>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Constant::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Constant>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::ConstantNaive::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == constane_naive_0_def);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "read_const_Constant_0_0");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "read_const_Constant_0_0test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
        EXPECT_TRUE(dep->required.count("header::fstream") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == constane_naive_0_def);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cuda") == 1);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::fstream") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::Constant, float>(0));
        cout << out[0].size() << endl;

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}
