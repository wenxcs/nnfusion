// Microsoft (c) 2019, Wenxiang
#include "constant_naive.hpp"

using namespace nnfusion::cuda;

ConstantNaive::ConstantNaive(ir::Constant_p inter_op)
    : CudaFunction(inter_op)
{
    assert_nullptr(inter_op) << "Constant Node is null.";
    assert_bool(inter_op->out.size() == 1) << "Constant Node only has one output.";
    const_name = inter_op->out[0].get_name();
    ofstream bin_file(const_name + ".bin", ios::out | ios::binary);
    NGRAPH_DEBUG << "Write Const [" << const_name << "] into file : " << const_name << ".bin ["
                 << inter_op->data_size << "] bytes" << endl;
    bin_file.write((const char*)inter_op->data_ptr, inter_op->data_size);
    bin_file.close();
    this->inter_op = inter_op;
}

string ConstantNaive::codegen_function_name()
{
    return "read_const_" + const_name;
}

string ConstantNaive::codegen_test_name()
{
    return codegen_function_name() + "test";
}

LanguageUnit_p ConstantNaive::codegen_function_definition()
{
    std::string name = codegen_function_name();
    create_ptr(LanguageUnit, lu, name);
    *lu << "extern \"C\" void " << name << "(" << inter_op->dtype << "** out)\n";
    (*lu).block_begin();
    {
        // Where should we put the planning code
        // *lu << "cudaMalloc((void**)out, " << inter_op->data_size << ")\n";
        *lu << "std::ifstream bin_file(\"" << const_name
            << ".bin\" , std::ios::in | std::ios::binary);\n"
            << "cudaMalloc((void**)out, " << inter_op->data_size << ");\n"
            << "char* tmp_mem = new char[" << inter_op->data_size << "];\n"
            << "bin_file.read(tmp_mem, " << inter_op->data_size << ");\n"
            << "cudaMemcpy(*out, tmp_mem, " << inter_op->data_size << ", cudaMemcpyHostToDevice);\n"
            << "bin_file.close();\n";
    }
    (*lu).block_end();
    return lu;
}

LanguageUnit_p ConstantNaive::codegen_function_call()
{
    std::string name = codegen_function_name() + "_call";
    create_ptr(LanguageUnit, lu, name);
    *lu << codegen_function_name() << "(&" << const_name << ");\n";
    return lu;
}

LanguageUnit_p ConstantNaive::codegen_dependency()
{
    shared_ptr<LanguageUnit> lu(new LanguageUnit);
    lu->require(header::cuda);
    lu->require(header::fstream);
    return lu;
}

CudaFunction_p ConstantNaive::codegen(ir::Operator_p inter_op)
{
    create_ptr(ConstantNaive, cop, static_pointer_cast<ir::Constant>(inter_op));
    NGRAPH_DEBUG << "Codegen for ConstantNaive:" << cop->codegen_function_name() << endl;
    return cop;
}