// Microsoft (c) 2019, Wenxiang
#pragma once
#include "../../op/constant.hpp"
#include "../cuda_function.hpp"
#include "../cuda_helper.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace cuda
    {
        class ConstantNaive : public CudaFunction
        {
        private:
            string const_name;
            ir::Constant_p inter_op;

        public:
            ConstantNaive(ir::Constant_p inter_op);
            string codegen_function_name() override;
            string codegen_test_name() override;

            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
            LanguageUnit_p codegen_dependency() override;

            static CudaFunction_p codegen(ir::Operator_p inter_op);
        };

        using ConstantNaive_p = shared_ptr<ConstantNaive>;
    }
}
