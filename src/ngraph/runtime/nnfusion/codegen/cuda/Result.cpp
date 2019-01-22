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
    return "cuda_result";
}

string cuda::Result::codegen_source_name()
{
    return "cuda_result.cu";
}

shared_ptr<CodeWriter> cuda::Result::codegen_function_definition()
{
    shared_ptr<CodeWriter> cw(new CodeWriter);
    CodeWriter& writer = *cw;
    writer << "// No codegen for Result since it's memcpy().\n";
    return cw;
}

shared_ptr<CodeWriter> cuda::Result::codegen_function_call()
{
    shared_ptr<CodeWriter> cw(new CodeWriter);
    CodeWriter& writer = *cw;
    assert_bool(inter_op->args.size() == 1) << "Input size mismatches.";
    assert_bool(inter_op->out.size() == 1) << "Output size mismatches.";
    emit_memcpyDtD(writer, inter_op->args[0], inter_op->out[0]);
    return cw;
}

shared_ptr<CodeWriter> cuda::Result::codegen_test()
{
    shared_ptr<CodeWriter> cw(new CodeWriter);
    CodeWriter& writer = *cw;
    writer << "// No test codegen for result OP\n";
    writer << "/*\n";
    writer << codegen_function_definition()->get_code();
    writer << codegen_function_call()->get_code();
    writer << "*/\n";
    return cw;
}

shared_ptr<CodeWriter> cuda::Result::codegen_dependency()
{
    shared_ptr<CodeWriter> cw(new CodeWriter);
    CodeWriter& writer = *cw;
    writer << "#include <cuda.h>\n";
    return cw;
}

std::shared_ptr<CodeGenOP> cuda::Result::codegen(std::shared_ptr<IntermediateOP> inter_op)
{
    shared_ptr<Result> cop(new Result(inter_op));
    NGRAPH_DEBUG << "Codegen for Result function:" << cop->codegen_function_name() << endl;
    return cop;
}