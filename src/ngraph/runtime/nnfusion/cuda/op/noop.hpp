// Microsoft (c) 2019, Wenxiang
#pragma once
#include "../cuda_function.hpp"
#include "../cuda_helper.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace cuda
    {
        class Noop : public CudaFunction
        {
        public:
            Noop(ir::Operator_p inter_op)
                : CudaFunction(inter_op)
            {
            }
            string codegen_function_name() override { return "cuda_noop"; }
            string codegen_test_name() override { return "cuda_noop_test"; }
            LanguageUnit_p codegen_function_definition() override
            {
                return LanguageUnit_p(new LanguageUnit);
            }
            LanguageUnit_p codegen_function_call() override
            {
                return LanguageUnit_p(new LanguageUnit);
            }
            LanguageUnit_p codegen_dependency() override
            {
                return LanguageUnit_p(new LanguageUnit);
            };

        public:
            static CudaFunction_p codegen(ir::Operator_p inter_op)
            {
                create_ptr(Noop, cop, inter_op);
                NGRAPH_DEBUG << "Codegen for Noop function:" << cop->codegen_function_name()
                             << endl;
                return cop;
            }
        };

        using Noop_p = shared_ptr<Noop>;
    }
}