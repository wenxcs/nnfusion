// Microsoft (c) 2019, Wenxiang
#pragma once
#include "cuda_helper.hpp"

namespace nnfusion
{
    namespace cuda
    {
        class CudaFunction : public ir::Function
        {
        public:
            CudaFunction() {}
            CudaFunction(ir::Operator_p inter_op)
                : Function(inter_op)
            {
            }

            LanguageUnit_p codegen_test() override;

            string gen_comments();
        };

        using CudaFunction_p = shared_ptr<CudaFunction>;
    }
}