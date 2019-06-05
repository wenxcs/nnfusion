// Microsoft (c) 2019, Wenxiang
#include "anyop.hpp"

cuda::Anyop::Anyop(ir::Operator_p inter_op)
    : CudaFunction(inter_op)
{
    enforce_not_nullptr(this->aop = static_pointer_cast<ir::Anyop>(inter_op));
}

string cuda::Anyop::codegen_function_name()
{
    std::stringstream kernel_name;
    kernel_name << "cuda"
                << "_anyop"
                << "_" << join(aop->dtypes, "_") << "_" << join(aop->dsizes, "_");

    return kernel_name.str();
}

string cuda::Anyop::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

LanguageUnit_p cuda::Anyop::codegen_function_definition()
{
    LanguageUnit_p cw(new LanguageUnit(codegen_function_name()));
    *cw << "void " << cw->symbol << "(";
    vector<string> params;
    for (int i = 0; i < aop->args.size(); i++)
        params.push_back(aop->dtypes[i] + "* in" + to_string(i));
    for (int i = 0; i < aop->out.size(); i++)
        params.push_back(aop->dtypes[i + aop->args.size()] + "* out" + to_string(i));
    *cw << join(params, ", ") << ")\n";
    (*cw).block_begin();
    *cw << "// No code for Anyop\n";
    (*cw).block_end();
    return cw;
}

LanguageUnit_p cuda::Anyop::codegen_function_call()
{
    LanguageUnit_p cw(new LanguageUnit(codegen_function_name() + "_call"));
    vector<string> names;
    names.insert(names.end(), aop->arg_names.begin(), aop->arg_names.end());
    names.insert(names.end(), aop->out_names.begin(), aop->out_names.end());
    *cw << codegen_function_name() << "( " << join(names, ", ") << ");\n";
    return cw;
}

LanguageUnit_p cuda::Anyop::codegen_dependency()
{
    LanguageUnit_p cw(new LanguageUnit(codegen_function_name() + "_dep"));
    return cw;
}

cuda::CudaFunction_p cuda::Anyop::codegen(ir::Operator_p inter_op)
{
    Anyop_p cop(new Anyop(inter_op));
    LOG_INFO << "Codegen for Anyop function:" << cop->codegen_function_name() << endl;
    std::cerr << "WARNING: using Any op for " << cop->op->node->get_friendly_name() << endl;
    return cop;
}