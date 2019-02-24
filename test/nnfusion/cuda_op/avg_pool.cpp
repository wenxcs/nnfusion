// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for ir::anyop
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/cuda/op/avg_pool.hpp"
#include "../test_util/common.hpp"

const static std::string avgpool_1_def =
    R"(extern "C" __global__ void cuda_avgpool_s2_2_14_r2_2_12_st1_ip0(float* in, float* out)
{
    float alpha = 1;
    float beta = 0;
    int N = 2;
    int C = 2;
    int D = 1;
    int H = 1;
    int W = 14;
    int HW = 14;
    int DHW = 14;
    int CDHW = 28;
    int magic_N = 1;
    int shift_N = 1;
    int P = 1;
    int Q = 12;
    int magic_P = 1;
    int shift_P = 0;
    int PQ = 12;
    int MPQ = 12;
    int KMPQ = 24;
    int S = 3;
    int RS = 3;
    int TRS = 3;
    int magic_S = -1431655765;
    int shift_S = 1;
    int magic_RS = -1431655765;
    int shift_RS = 1;
    int str_d = 0;
    int str_h = 0;
    int str_w = 1;
    int pad_d = 0;
    int pad_h = 0;
    int pad_w = 0;
    const int tid = threadIdx.x;
    if (tid < 32)
    {
        const int q = blockIdx.x;
        const int mp = blockIdx.y;
        const int nk = blockIdx.z;
        const int k = division_by_invariant_multiplication(nk, magic_N, shift_N);
        const int n = nk - k * N;
        const int m = division_by_invariant_multiplication(mp, magic_P, shift_P);
        const int p = mp - m * P;
        out += n*KMPQ + k*MPQ + m*PQ + mad16(p, Q, q);
        int qs = q * str_w - pad_w;
        int pr = p * str_h - pad_h;
        int mt = m * str_d - pad_d;
        int pool_size = 0;
        float sum = 0.0f;
        float rcp_pool_size = 1.0f;
        for (int trs = tid; trs < TRS; trs += 32)
        {
            int t = division_by_invariant_multiplication(trs, magic_RS, shift_RS);
            int rs = mod16(trs, t, RS);
            int r  = division_by_invariant_multiplication(rs, magic_S, shift_S);
            int s  = mod16(rs, r, S);
            int x = qs + s;
            int y = pr + r;
            int z = mt + t;
            bool bounds_x = (x >= 0) && (x < W);
            bool bounds_y = (y >= 0) && (y < H);
            bool bounds_z = (z >= 0) && (z < D);
            bool within_tensor_bounds = bounds_x && bounds_y && bounds_z;
            pool_size += __popc(__ballot_sync(0xffffffff, within_tensor_bounds));
            int idx = n*CDHW + k*DHW + z*HW + y*W + x;
            sum += load(in,idx,within_tensor_bounds);
        }
        rcp_pool_size = 1.0f / (float)pool_size;
        for (int i = 16; i > 0; i >>= 1)
        {
            sum += __shfl_xor_sync(0xffffffff,sum,i,32);
        }
        if (tid == 0)
        {
            *out = sum * rcp_pool_size;
        }
    }
}
)";

const static std::string avgpool_0_def =
    R"(void cudnn_avgpool_dtype_float_i1_1_3_3_o1_1_4_4_ws2_2_wst1_1_pb1_1_pb1_1(cudnnHandle_t cudnn_handle, float* in, float* out)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 3, 3));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 4, 4));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,2, 2, 1, 1, 1, 1));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, in, &beta, output_desc, out));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));
}
)";

TEST(nnfusion_cuda_op, avgpool_1d)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::AvgPool>(1);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::AvgPool::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::AvgPool>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::AvgPool::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == avgpool_1_def);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cuda_avgpool_s2_2_14_r2_2_12_st1_ip0");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname == "cuda_avgpool_s2_2_14_r2_2_12_st1_ip0_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cuda") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == avgpool_1_def);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cuda") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        /*intput*/ in.push_back(nnfusion::inventory::generate_input<op::AvgPool, float>(1));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::AvgPool, float>(1));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}

TEST(nnfusion_cuda_op, avgpool_md)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::AvgPool>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::AvgPool::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::AvgPool>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::AvgPool::codegen(op);

    { // Test methods
        // Check generated function definition:
        auto def = cuda_op->codegen_function_definition();
        EXPECT_TRUE(def->get_code() == avgpool_0_def);
        // Check function call
        auto call = cuda_op->codegen_function_call();
        EXPECT_TRUE(call->get_code().size() > 0);
        // Check function name
        auto name = cuda_op->codegen_function_name();
        EXPECT_TRUE(name == "cudnn_avgpool_dtype_float_i1_1_3_3_o1_1_4_4_ws2_2_wst1_1_pb1_1_pb1_1");
        auto testname = cuda_op->codegen_test_name();
        EXPECT_TRUE(testname ==
                    "cudnn_avgpool_dtype_float_i1_1_3_3_o1_1_4_4_ws2_2_wst1_1_pb1_1_pb1_1_test");
        auto dep = cuda_op->codegen_dependency();
        EXPECT_TRUE(dep->required.count("header::cudnn") == 1);
    }

    { // Test codegen procedure
        auto test = cuda_op->codegen_source();
        EXPECT_TRUE(cuda_op->definition_unit->get_code() == avgpool_0_def);
        EXPECT_TRUE(cuda_op->call_unit->get_code().size() > 0);
        EXPECT_TRUE(cuda_op->dep_unit->required.count("header::cudnn") == 1);

        nnfusion::library::dump_test_code(cuda_op);

        vector<vector<float>> in;
        /*intput*/ in.push_back(nnfusion::inventory::generate_input<op::AvgPool, float>(0));
        vector<vector<float>> out;
        out.push_back(nnfusion::inventory::generate_output<op::AvgPool, float>(0));

        auto result = nnfusion::library::execute_op(cuda_op->codegen_test_name(), in, out);
        EXPECT_TRUE(ngraph::test::all_close_f(out[0], result[0]));
    }
}