// Microsoft(c)2019, NNFusion Team

#include "codegenerator_helper.hpp"

#include "nnfusion/core/kernels/cpu/cpu_langunit.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/engine/async_manager.hpp"

using namespace nnfusion;
using namespace nnfusion::codegenerator;
using namespace nnfusion::kernels;
using namespace nnfusion::async;

LanguageUnit_p extern_function(LanguageUnit_p lu)
{
}

LanguageUnit_p extern_variable(LanguageUnit_p lu)
{
}

void FunctionFile::save_file()
{
    LanguageUnit def_re("require");
    def_re << "// Microsoft (c) 2019, NNFusion\n";

    // Write Dependency
    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("header::") != string::npos)
            def_re << it.second->get_code();
    def_re << "\n";

    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("macro::") != string::npos)
            def_re << it.second->get_code();
    def_re << "\n";
    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("declaration::") != string::npos)
        {
            if (it.second->get_symbol() == "declaration::global_cublas_handle" ||
                it.second->get_symbol() == "declaration::global_cudnn_handle" ||
                it.second->get_symbol() == "declaration::num_SMs" ||
                it.second->get_symbol() == "declaration::allreduce_stream" ||
                it.second->get_symbol() == "declaration::applygradient_stream")
            {
                def_re << "extern ";
            }
            def_re << it.second->get_code();
        }
    def_re << "\n";

    string fname = this->get_symbol();
    if (fname.length() > 128)
    {
        size_t hashcode = std::hash<std::string>{}(fname);
        fname = "kernels/compressed_src_" + std::to_string(hashcode) + suffix_str;
    }
    else
        fname = "kernels/" + this->get_symbol() + suffix_str;

    ofstream src(fname);
    src << def_re.get_code();
    src << "\n";
    src << this->get_code();
    src.close();
}

void FunctionFile::merge_from(FunctionFile_p func)
{
    // Merge required symbols;
    for (auto& sym : func->local_symbol)
        require(sym.second);
    // Merge source code;
    (*this) << "\n" << func->get_code();
    // Merge symbol name;
    if (get_symbol() == "")
        change_symbol(func->get_symbol());
    else
        change_symbol(get_symbol() + "_" + func->get_symbol());
    extern_declare = extern_declare + "\n" + func->extern_declare;
}

FunctionFile::FunctionFile(string extern_declare, LanguageUnit_p file_context)
{
    // Get requirement
    this->clean_require();
    for (auto& sym : file_context->local_symbol)
        this->require(sym.second);
    // Get source code
    (*this) << file_context->get_code();
    change_symbol(file_context->get_symbol());
    this->extern_declare = extern_declare;
}

FunctionFile_p FunctionFile::convert_from(std::shared_ptr<nnfusion::kernels::KernelEmitter> kernel)
{
    FunctionUnit_p fu = kernel->get_or_emit_source();
    LanguageUnit_p lu = make_shared<LanguageUnit>();
    LanguageUnit& def = *lu;
    def.require(header::assert);
    def.require(header::stdexcept);
    def.require(header::sstream);
    def.require(header::cuda);
    def.require(header::cublas);
    def.require(header::cudnn);
    def.require(macro::CUDA_SAFE_CALL);
    def.require(macro::CUDNN_SAFE_CALL);
    def.require(macro::CUBLAS_SAFE_CALL);
    def.require(declaration::typedef_int);

    for (auto& sym : fu->dep_unit->local_symbol)
        def.require(sym.second);

    string body_unit = fu->body_unit->get_code();
    std::string sig = fu->get_specialized_signature();
    int handle_cudnn = body_unit.find("cudnn_handle");
    int handle_cublas = body_unit.find("cublas_handle");
    auto gnode = kernel->m_context->gnode;
    auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();

    // conv kernels in the the stream shares the same workspace_ptr
    if (gnode->get_op_type() == "Convolution")
    {
        std::string s_workspace =
            "workspace_ptr_" + to_string(async_info.execution_stream->get_stream_id());
        int pos = body_unit.find("workspace_ptr");
        while (pos >= 0)
        {
            body_unit.replace(pos, 13, s_workspace);
            pos = body_unit.find("workspace_ptr", pos + s_workspace.size());
        }
    }

    // add stream or handle parameter for cuda lib kernel
    int pos = sig.find("(");
    if (handle_cudnn >= 0)
        sig.insert(pos + 1, "cudnnHandle_t cudnn_handle, ");

    if (handle_cublas >= 0)
        sig.insert(pos + 1, "cublasHandle_t cublas_handle, ");

    if (fu->body_unit->get_code().find("stream") != string::npos)
        sig.insert(pos + 1, "cudaStream_t stream, ");

    // This for cudalib call or __global__ functions;
    def << fu->comment_unit->get_code();
    def << sig << "\n";
    def.block_begin();
    def << body_unit << "\n";
    def.block_end();

    LanguageUnit dec("dec");
    {
        if (sig.find("extern ") != 0)
            sig = "extern " + sig;
        dec << "\n" << sig << ";\n";
    }

    string sname = fu->name_unit->get_code();
    def.change_symbol(sname);
    return make_shared<FunctionFile>(dec.get_code(), lu);
}

CPUFunctionFile_p
    CPUFunctionFile::convert_from(std::shared_ptr<nnfusion::kernels::KernelEmitter> kernel)
{
    FunctionUnit_p fu = kernel->get_or_emit_source();
    LanguageUnit_p lu = make_shared<LanguageUnit>();
    LanguageUnit& def = *lu;
    def.require(header::assert);
    def.require(header::stdexcept);
    def.require(header::sstream);
    def.require(header::fstream);
    def.require(header::thread);

    if (kernel->is_parallelism())
        def.require(header::threadpool);

    for (auto& sym : fu->dep_unit->local_symbol)
        def.require(sym.second);

    def << fu->comment_unit->get_code();
    string sig = fu->get_specialized_signature();

    if (kernel->is_parallelism())
    {
        int pos = sig.find("(");
        if (pos >= 0)
        {
            sig.insert(pos + 1, "concurrency::ThreadPool* thread_pool, ");
        }
    }

    def << sig << "\n";
    def.block_begin();
    def << fu->body_unit->get_code() << "\n";
    def.block_end();

    LanguageUnit dec("dec");
    {
        if (sig.find("extern ") != 0)
            sig = "extern " + sig;
        dec << "\n" << sig << ";\n";
    }

    string sname = fu->name_unit->get_code();
    def.change_symbol(sname);
    return make_shared<CPUFunctionFile>(dec.get_code(), lu);
}

void CPUFunctionFile::save_file()
{
    LanguageUnit def_re("require");
    def_re << "// Microsoft (c) 2019, NNFusion\n";

    // Write Dependency
    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("header::") != string::npos)
        {
            if (it.second->symbol.find("header::reference_common") != string::npos)
            {
                def_re << R"(
#include "../reference_common.h"
using namespace reference_common;
)";
            }
            else
            {
                def_re << it.second->get_code();
            }
        }
    def_re << "#include<cstring>\n";
    def_re << "using namespace std;\n";
    def_re << "\n";

    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("macro::") != string::npos)
            def_re << it.second->get_code();
    def_re << "\n";
    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("declaration::") != string::npos)
            def_re << it.second->get_code();
    def_re << "\n";
    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("cpu_reference_") != string::npos)
            def_re << it.second->get_code();
    def_re << "\n";

    string fname = this->get_symbol();
    if (fname.length() > 128)
    {
        size_t hashcode = std::hash<std::string>{}(fname);
        fname = "kernels/compressed_src_" + std::to_string(hashcode) + suffix_str;
    }
    else
        fname = "kernels/" + this->get_symbol() + suffix_str;

    ofstream src(fname);
    src << def_re.get_code();
    src << "\n";
    src << this->get_code();
    src.close();
}