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
                        Result(shared_ptr<IntermediateOP> inter_op)
                            : CodeGenOP(inter_op)
                        {
                        }

                        string codegen_function_name() override { return "result"; }
                        string codegen_source_name() override { return "result.cu"; }
                        shared_ptr<CodeWriter> codegen_function_definition() override
                        {
                            shared_ptr<CodeWriter> cw(new CodeWriter);
                            CodeWriter& writer = *cw;
                            writer << "// Function Body\n";
                            return cw;
                        }

                        shared_ptr<CodeWriter> codegen_function_call() override
                        {
                            shared_ptr<CodeWriter> cw(new CodeWriter);
                            CodeWriter& writer = *cw;
                            writer << "// Function Call\n";
                            return cw;
                        }

                        shared_ptr<CodeWriter> codegen_test() override
                        {
                            shared_ptr<CodeWriter> cw(new CodeWriter);
                            CodeWriter& writer = *cw;
                            writer << "// Function Test\n";
                            return cw;
                        }

                        shared_ptr<CodeWriter> codegen_dependency() override
                        {
                            shared_ptr<CodeWriter> cw(new CodeWriter);
                            CodeWriter& writer = *cw;
                            writer << "// Function Includes\n";
                            return cw;
                        }

                    public:
                        static std::shared_ptr<CodeGenOP>
                            codegen(std::shared_ptr<IntermediateOP> inter_op)
                        {
                            shared_ptr<Result> cop(new Result(inter_op));
                            NGRAPH_DEBUG
                                << "Codegen for Result function:" << cop->codegen_function_name()
                                << endl;
                            return cop;
                        }
                    };
                }
            }
        }
    }
}