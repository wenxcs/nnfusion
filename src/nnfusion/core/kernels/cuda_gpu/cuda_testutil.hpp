// Microsoft (c) 2019, Wenxiang
#pragma once
#include "cuda_helper.hpp"

using CodeWriter = ngraph::codegen::CodeWriter;

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            void test_cudaMemcpyDtoH(CodeWriter& writer, const TensorWrapper tensor);
            void test_cudaMemcpyHtoD(CodeWriter& writer, const TensorWrapper tensor);
            void test_cudaMalloc(CodeWriter& writer, const TensorWrapper tensor);
            vector<float> test_hostData(CodeWriter& writer, const TensorWrapper tensor);
            vector<float>
                test_hostData(CodeWriter& writer, const TensorWrapper tensor, vector<float>& d);
            void test_compare(CodeWriter& writer, const TensorWrapper tensor);
        }
    }
}
