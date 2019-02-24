// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for ir::anyop
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/cuda/op/batchnorm.hpp"
#include "../test_util/common.hpp"

const static std::string bn_0_call =
    R"(void cudnn_bn_op_dtype_float_ts2_2_2_1_ps2(cudnnHandle_t cudnn_handle, float* in, float* out, void* gain, void* bias, void* mean, void* variance, double eps)
{
    cudnnTensorDescriptor_t tensor_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&tensor_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 2, 2, 1));
    cudnnTensorDescriptor_t derived_param_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&derived_param_desc));
    CUDNN_SAFE_CALL(cudnnDeriveBNTensorDescriptor(derived_param_desc, tensor_desc, CUDNN_BATCHNORM_SPATIAL));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnBatchNormalizationForwardInference(cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, tensor_desc, in, tensor_desc, out, derived_param_desc, gain, bias, mean, variance, eps));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(derived_param_desc));
}
)";

TEST(nnfusion_cuda_op, BatchNorm)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::BatchNormInference>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::BatchNorm::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::BatchNorm>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::BatchNorm::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == bn_0_call);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cudnn_bn_op_dtype_float_ts2_2_2_1_ps2");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cudnn_bn_op_dtype_float_ts2_2_2_1_ps2_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cudnn") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == bn_0_call);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cudnn") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        auto param = nnfusion::inventory::generate_param<op::BatchNormInference, float>(0);

        vector<vector<float>> in;
        in.push_back(vector<float>(param.begin(), param.begin() + 2));
        in.push_back(vector<float>(param.begin() + 2, param.begin() + 4));
        in.push_back(nnfusion::inventory::generate_input<op::BatchNormInference, float>(0));
        in.push_back(vector<float>(param.begin() + 4, param.begin() + 6));
        in.push_back(vector<float>(param.begin() + 6, param.begin() + 8));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::BatchNormInference, float>(0));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}