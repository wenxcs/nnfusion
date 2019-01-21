// Microsoft (c) 2019, Wenxiang
#pragma once
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_op.hpp"

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
                    class Result : public CodeGenOP
                    {
                    public:
                        Result(shared_ptr<IntermediateOP> inter_op);
                        string codegen_function_name() override;
                        string codegen_source_name() override;
                        shared_ptr<CodeWriter> codegen_function_definition() override;
                        shared_ptr<CodeWriter> codegen_function_call() override;
                        shared_ptr<CodeWriter> codegen_test() override;
                        shared_ptr<CodeWriter> codegen_dependency() override;
                        static std::shared_ptr<CodeGenOP>
                            codegen(std::shared_ptr<IntermediateOP> inter_op);
                    };
                }
            }
        }
    }
}