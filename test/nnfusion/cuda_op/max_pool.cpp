// Microsoft (c) 2019, Yuchao
/**
 * \brief Unit tests for MaxPool
 * \author Yuchao Zheng
 */

#include "ngraph/runtime/nnfusion/cuda/op/max_pool.hpp"
#include "../test_util/common.hpp"

const static std::string maxpool_0_def =
    R"(void cudnn_maxpool_dtype_float_i1_1_3_3_o1_1_4_4_ws2_2_wst1_1_pb1_1_pb1_1(cudnnHandle_t cudnn_handle, float* in, float* out)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 3, 3));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 4, 4));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,2, 2, 1, 1, 1, 1));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, in, &beta, output_desc, out));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));
}
)";

const static std::string maxpool_1_def =
    R"(extern "C" __global__ void cuda_maxpool_float_float_iw14_ow12_ww3_wst1(float* in, float* out, size_t nthreads)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nthreads)
    {
        size_t start = (tid / 12) * 14 +  (tid % 12) * 1;
        float max_val = -3.4028235e+38;
        for (size_t i = start; i < start + 3; i++)
        {
            const float input = in[i];
            if (input > max_val)
            {
                max_val = input;
            }
        }
        out[tid] = max_val;
    }
}
)";

TEST(nnfusion_cuda_op, maxpool_1d)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::MaxPool>(1);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::MaxPool::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::MaxPool>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::MaxPool::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == maxpool_1_def);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cuda_maxpool_float_float_iw14_ow12_ww3_wst1");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cuda_maxpool_float_float_iw14_ow12_ww3_wst1_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == maxpool_1_def);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cuda") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        /*intput*/ in.push_back(nnfusion::inventory::generate_input<op::MaxPool, float>(1));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::MaxPool, float>(1));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}

TEST(nnfusion_cuda_op, maxpool_md)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::MaxPool>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::MaxPool::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::MaxPool>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::MaxPool::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == maxpool_0_def);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cudnn_maxpool_dtype_float_i1_1_3_3_o1_1_4_4_ws2_2_wst1_1_pb1_1_pb1_1");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname ==
                    "cudnn_maxpool_dtype_float_i1_1_3_3_o1_1_4_4_ws2_2_wst1_1_pb1_1_pb1_1_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cudnn") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == maxpool_0_def);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cudnn") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        /*intput*/ in.push_back(nnfusion::inventory::generate_input<op::MaxPool, float>(0));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::MaxPool, float>(0));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}