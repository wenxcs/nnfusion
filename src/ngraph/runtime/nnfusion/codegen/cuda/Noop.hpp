// Microsoft (c) 2019, Wenxiang
#pragma once
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_codegenop.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_helper.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_langunit.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"

using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            namespace codegen
            {
                namespace cuda
                {
                    class Noop : public CudaCodeGenOP
                    {
                    public:
                        Noop(shared_ptr<IntermediateOP> inter_op) { this->inter_op = inter_op; }
                        string codegen_function_name() override { return "cuda_noop"; }
                        string codegen_test_name() override { return "cuda_noop_test"; }
                        shared_ptr<LanguageUnit> codegen_function_definition() override
                        {
                            return shared_ptr<LanguageUnit>(new LanguageUnit);
                        }
                        shared_ptr<LanguageUnit> codegen_function_call() override
                        {
                            return shared_ptr<LanguageUnit>(new LanguageUnit);
                        }
                        shared_ptr<LanguageUnit> codegen_dependency() override
                        {
                            return shared_ptr<LanguageUnit>(new LanguageUnit);
                        };

                    public:
                        static std::shared_ptr<CodeGenOP>
                            codegen(std::shared_ptr<IntermediateOP> inter_op)
                        {
                            shared_ptr<Noop> cop(new Noop(inter_op));
                            NGRAPH_DEBUG
                                << "Codegen for Noop function:" << cop->codegen_function_name()
                                << endl;
                            return cop;
                        }
                    };
                }
            }
        }
    }
}