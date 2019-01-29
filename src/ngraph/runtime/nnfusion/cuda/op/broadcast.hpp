// Microsoft (c) 2019, Wenxiang
#pragma once
#include "../cuda_function.hpp"
#include "../cuda_helper.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace cuda
    {
        class Broadcast : public CudaFunction
        {
        public:
            ir::Broadcast_p _op;

        public:
            Broadcast(ir::Operator_p inter_op);
            string codegen_function_name() override;
            string codegen_test_name() override;
            LanguageUnit_p codegen_function_definition() override;
            LanguageUnit_p codegen_function_call() override;
            LanguageUnit_p codegen_dependency() override;
            static CudaFunction_p codegen(ir::Operator_p inter_op);
        };

        using Broadcast_p = shared_ptr<Broadcast>;
    }
}