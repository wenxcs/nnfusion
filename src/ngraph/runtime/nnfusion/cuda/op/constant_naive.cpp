// Microsoft (c) 2019, Wenxiang
#include "constant_naive.hpp"

#include <bits/stdc++.h>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace nnfusion::cuda;

static bool create_dir(std::string tar_path)
{
    bool flag;
    int mkdir_status;
    struct stat s;
    int err = stat(tar_path.c_str(), &s);
    if (-1 == err)
    {
        mkdir_status = mkdir((tar_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == mkdir_status)
        {
            printf("Error creating directory: %s", (tar_path).c_str());
            flag = false;
        }
        else
            flag = true;
    }
    else
    {
        flag = true;
    }
    return flag;
}

ConstantNaive::ConstantNaive(ir::Constant_p inter_op)
    : CudaFunction(inter_op)
{
    enforce_not_nullptr(inter_op) << "Constant Node is null.";
    enforce(inter_op->out.size() == 1) << "Constant Node only has one output.";
    folder = "./Constant/";
    create_dir(folder);
    const_name = inter_op->out[0].get_name();
    ofstream bin_file(folder + const_name + ".bin", ios::out | ios::binary);
    LOG_INFO << "Write Const [" << const_name << "] into file : " << const_name << ".bin ["
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
        *lu << "std::ifstream bin_file(\"" << folder + const_name
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
    LOG_INFO << "Codegen for ConstantNaive:" << cop->codegen_function_name() << endl;
    return cop;
}