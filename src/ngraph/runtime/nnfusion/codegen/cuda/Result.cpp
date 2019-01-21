// Microsoft (c) 2019, Wenxiang
#include "ngraph/runtime/nnfusion/codegen/cuda/Result.hpp"

using namespace ngraph;
using namespace ngraph::runtime::nnfusion::codegen;

cuda::Result::Result(shared_ptr<IntermediateOP> inter_op)
    : CodeGenOP(inter_op)
{
}

string cuda::Result::codegen_function_name()
{
    return "result";
}

string cuda::Result::codegen_source_name()
{
    return "result.cu";
}

shared_ptr<CodeWriter> cuda::Result::codegen_function_definition()
{
    shared_ptr<CodeWriter> cw(new CodeWriter);
    CodeWriter& writer = *cw;
    writer << "// Function Body\n";
    return cw;
}

shared_ptr<CodeWriter> cuda::Result::codegen_function_call()
{
    shared_ptr<CodeWriter> cw(new CodeWriter);
    CodeWriter& writer = *cw;
    writer << "// Function Call\n";
    return cw;
}

shared_ptr<CodeWriter> cuda::Result::codegen_test()
{
    shared_ptr<CodeWriter> cw(new CodeWriter);
    CodeWriter& writer = *cw;
    writer << "// Function Test\n";
    return cw;
}

shared_ptr<CodeWriter> cuda::Result::codegen_dependency()
{
    shared_ptr<CodeWriter> cw(new CodeWriter);
    CodeWriter& writer = *cw;
    writer << "// Function Includes\n";
    return cw;
}

std::shared_ptr<CodeGenOP> cuda::Result::codegen(std::shared_ptr<IntermediateOP> inter_op)
{
    shared_ptr<Result> cop(new Result(inter_op));
    NGRAPH_DEBUG << "Codegen for Result function:" << cop->codegen_function_name() << endl;
    return cop;
}