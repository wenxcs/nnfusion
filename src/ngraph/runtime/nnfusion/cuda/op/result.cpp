// Microsoft (c) 2019, Wenxiang
#include "result.hpp"

cuda::Result::Result(ir::Operator_p inter_op)
    : CudaFunction(inter_op)
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

LanguageUnit_p cuda::Result::codegen_function_definition()
{
    LanguageUnit_p cw(new LanguageUnit);
    LanguageUnit& writer = *cw;
    writer << "// No codegen for Result since it's memcpy().\n";
    return cw;
}

LanguageUnit_p cuda::Result::codegen_function_call()
{
    LanguageUnit_p cw(new LanguageUnit);
    LanguageUnit& writer = *cw;
    assert_bool(op->args.size() == 1) << "Input size mismatches.";
    assert_bool(op->out.size() == 1) << "Output size mismatches.";
    emit_memcpyDtD(writer, op->out[0], op->args[0]);
    return cw;
}

LanguageUnit_p cuda::Result::codegen_dependency()
{
    LanguageUnit_p cw(new LanguageUnit);
    cw->require(header::cuda);
    return cw;
}

cuda::CudaFunction_p cuda::Result::codegen(ir::Operator_p inter_op)
{
    Result_p cop(new Result(inter_op));
    NGRAPH_DEBUG << "Codegen for Result function:" << cop->codegen_function_name() << endl;
    return cop;
}