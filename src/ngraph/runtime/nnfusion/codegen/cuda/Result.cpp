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

string cuda::Result::codegen_test_name()
{
    return "cuda_result_test";
}

shared_ptr<LanguageUnit> cuda::Result::codegen_function_definition()
{
    shared_ptr<LanguageUnit> cw(new LanguageUnit);
    LanguageUnit& writer = *cw;
    writer << "// No codegen for Result since it's memcpy().\n";
    return cw;
}

shared_ptr<LanguageUnit> cuda::Result::codegen_function_call()
{
    shared_ptr<LanguageUnit> cw(new LanguageUnit);
    LanguageUnit& writer = *cw;
    assert_bool(inter_op->args.size() == 1) << "Input size mismatches.";
    assert_bool(inter_op->out.size() == 1) << "Output size mismatches.";
    emit_memcpyDtD(writer, inter_op->args[0], inter_op->out[0]);
    return cw;
}

shared_ptr<LanguageUnit> cuda::Result::codegen_test()
{
    shared_ptr<LanguageUnit> cw(new LanguageUnit);
    LanguageUnit& writer = *cw;
    writer << "// No test codegen for result OP\n";
    return cw;
}

shared_ptr<LanguageUnit> cuda::Result::codegen_test_call()
{
    shared_ptr<LanguageUnit> cw(new LanguageUnit);
    LanguageUnit& writer = *cw;
    writer << "// No test for result OP\n";
    return cw;
}

shared_ptr<LanguageUnit> cuda::Result::codegen_dependency()
{
    shared_ptr<LanguageUnit> cw(new LanguageUnit);
    cw->require(shared_ptr<LanguageUnit>(new LanguageUnit("header_cuda_h", "#include <cuda.h>\n")));
    return cw;
}

std::shared_ptr<CodeGenOP> cuda::Result::codegen(std::shared_ptr<IntermediateOP> inter_op)
{
    shared_ptr<Result> cop(new Result(inter_op));
    NGRAPH_DEBUG << "Codegen for Result function:" << cop->codegen_function_name() << endl;
    return cop;
}