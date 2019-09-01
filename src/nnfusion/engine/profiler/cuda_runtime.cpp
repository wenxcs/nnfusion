// Microsoft (c) 2019, MSRA/NNFUSION Team
/**
 * \brief Profiler::CudaRuntime for creating a Compiler to profile the kernel
 * \author wenxh
 */

#include "cuda_runtime.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"

using namespace nnfusion::profiler;
using namespace nnfusion::kernels;

bool CudaDefaultRuntime::codegen(const ProfilingContext::Pointer& ke)
{
    if (ke->source_code != nullptr)
        return true;
    FunctionUnit_p fu = ke->kernel->get_or_emit_source();
    LanguageUnit writer(fu->name_unit->get_code() + ".cu");
    writer << "// Microsoft (c) 2019, NNFusion\n";

    auto re = fu->dep_unit;
    re->require(header::assert);
    re->require(header::stdexcept);
    re->require(header::sstream);
    re->require(header::cuda);
    re->require(macro::CUDA_SAFE_CALL);
    re->require(declaration::typedef_int);

    // Write Dependency
    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("header::") != string::npos)
            writer << it.second->get_code();
    writer << "\n";

    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("macro::") != string::npos)
            writer << it.second->get_code();
    writer << "\n";
    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("declaration::") != string::npos)
            writer << it.second->get_code();
    writer << "\n";

    // Write function definition
    writer << fu->comment_unit->get_code();
    writer << fu->get_specialized_signature() << "\n";
    writer.block_begin();
    writer << fu->body_unit->get_code() << "\n";
    writer.block_end();

    // Write Test Calls
    // extern "C" void cuda_some_op_test(type* in0, ..., type* out0, ....)
    //{
    //   call_global_func<<<(1, 1, 1), (1, 1, 1), 0, 0>>(in0, ..., out0, ...)
    //}

    auto& arg = ke->kernel->m_context->inputs;
    auto& out = ke->kernel->m_context->outputs;

    writer << "extern \"C\" double " << fu->name_unit->get_code() << "_host(";
    for (size_t i = 0; i + 1 < arg.size(); i++)
    {
        writer << arg[i].get_type() << "* " << arg[i].get_name() << "_host, ";
    }
    if (!arg.empty())
    {
        writer << arg.back().get_type() << "* " << arg.back().get_name();
        if (!out.empty())
            writer << "_host, ";
    }

    for (size_t i = 0; i + 1 < out.size(); i++)
    {
        writer << out[i].get_type() << "* " << out[i].get_name() << "_host, ";
    }
    if (!out.empty())
    {
        writer << out.back().get_type() << "* " << out.back().get_name() << "_host";
    }
    writer << ")\n";

    writer.block_begin();
    {
        if (re->local_symbol.count("declaration::global_cublas_handle") > 0)
        {
            writer << "CUBLAS_SAFE_CALL(cublasCreate(&global_cublas_handle));\n";
        }

        if (re->local_symbol.count("declaration::global_cudnn_handle") > 0)
        {
            writer << "CUDNN_SAFE_CALL(cudnnCreate(&global_cudnn_handle));\n";
        }

        if (re->local_symbol.count("declaration::num_SMs") > 0)
        {
            writer << "CUDA_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, "
                      "cudaDevAttrMultiProcessorCount, 0));\n";
        }

        for (size_t i = 0; i < arg.size(); i++)
        {
            auto& tensor = arg[i];
            writer << tensor.get_type() << "* " << tensor.get_name() << ";\n"
                   << "cudaMalloc((void**)&" << tensor.get_name() << "," << tensor.get_size()
                   << " * " << tensor.get_element_type().size() << ");\n";

            writer << "cudaMemcpy(" << tensor.get_name() << ", " << tensor.get_name() << "_host, "
                   << tensor.get_size() << " * " << tensor.get_element_type().size() << ", "
                   << "cudaMemcpyHostToDevice);\n";
        }
        for (size_t i = 0; i < out.size(); i++)
        {
            auto& tensor = out[i];
            writer << tensor.get_type() << "* " << tensor.get_name() << ";\n"
                   << "cudaMalloc((void**)&" << tensor.get_name() << "," << tensor.get_size()
                   << " * " << tensor.get_element_type().size() << ");\n";
        }

        writer << "cudaEvent_t start, stop;\n";
        writer << "cudaEventCreate(&start);\n";
        writer << "cudaEventCreate(&stop);\n";

        writer << "for(int i=0; i < " << ke->warmup_times + ke->runtime_times << "; i++)\n";
        writer.block_begin();
        {
            writer << "if(i == " << ke->warmup_times << ") cudaEventRecord(start);\n";
            writer << fu->name_unit->get_code() << fu->call_unit->get_code();
        }
        writer.block_end();

        writer << "cudaEventRecord(stop);\n";
        writer << "cudaEventSynchronize(stop);\n";
        writer << "float milliseconds = 0;\n";
        writer << "cudaEventElapsedTime(&milliseconds, start, stop);\n";

        for (size_t i = 0; i < out.size(); i++)
        {
            auto& tensor = out[i];
            writer << "cudaMemcpy(" << tensor.get_name() << "_host, " << tensor.get_name() << ", "
                   << tensor.get_size() << " * " << tensor.get_element_type().size() << ", "
                   << "cudaMemcpyDeviceToHost);\n";
        }

        for (size_t i = 0; i < arg.size(); i++)
        {
            auto& tensor = arg[i];
            writer << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
        }

        for (size_t i = 0; i < out.size(); i++)
        {
            auto& tensor = out[i];
            writer << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
        }

        if (re->local_symbol.count("declaration::global_cublas_handle") > 0)
        {
            writer << "CUBLAS_SAFE_CALL(cublasDestroy(global_cublas_handle));\n";
        }

        if (re->local_symbol.count("declaration::global_cudnn_handle") > 0)
        {
            writer << "CUDNN_SAFE_CALL(cudnnDestroy(global_cudnn_handle));\n";
        }
        writer << "return milliseconds/" << ke->runtime_times << ";\n";
    }
    writer.block_end();

    writer << "\n";

    writer << "extern \"C\" double " << fu->name_unit->get_code()
           << "_entry(void** args, void** outputs)";

    writer.block_begin();
    {
        writer << "return " << fu->name_unit->get_code() << "_host(";
        for (size_t i = 0; i + 1 < arg.size() + out.size(); i++)
        {
            string type = i < arg.size()
                              ? arg[i].get_type()
                              : (i - arg.size() < out.size() ? out[i - arg.size()].get_type() : "");
            writer << "(" << type << "*)" << (i < arg.size() ? "args" : "outputs") << "["
                   << i - (i >= arg.size() ? arg.size() : 0) << "], ";
        }
        if (arg.size() + out.size() > 0)
        {
            int i = arg.size() + out.size() - 1;
            string type = i < arg.size()
                              ? arg[i].get_type()
                              : (i - arg.size() < out.size() ? out[i - arg.size()].get_type() : "");
            writer << "(" << type << "*)" << (out.size() == 0 ? "args" : "outputs") << "["
                   << i - (i >= arg.size() ? arg.size() : 0) << "]";
        }
        writer << ");\n";
    }
    writer.block_end();

    ke->source_code = make_shared<LanguageUnit>(move(writer));
    return true;
    /*
    // Save the file
    ofstream source_file(writer.symbol);
    source_file << writer.get_code();
    source_file.close();
    return false;
    */
}

bool CudaDefaultRuntime::compile(const ProfilingContext::Pointer& ke)
{
    if (ke->entry_point != nullptr)
        return true;
    string filename = string(tmpnam(nullptr));
    string objname = filename + DLIB_SUFFIX;
    string srcname = filename + ".cu";
    // ofstream source_file(ke->working_dir + "/" + ke->source_code->symbol);
    ofstream source_file(srcname);
    source_file << ke->source_code->get_code();
    source_file.close();

    int ret = system(("nvcc\t--compiler-options\t'-fPIC\t-lcudnn\t-lcublas\t-lcudart'\t--shared\t" +
                      srcname + "\t-o\t" + objname)
                         .c_str());
    if (ret != 0)
        return false;
    if (!file_exsits(objname))
        return false;
    auto obj = get_library_handle(objname);
    auto entry = get_funcion_pointer(
        ke->kernel->get_or_emit_source()->name_unit->get_code() + "_entry", obj);
    if (entry == nullptr)
        return false;
    ke->entry_point = (double (*)(void**, void**))entry;
    return true;
}

double CudaDefaultRuntime::execute(const ProfilingContext::Pointer& ke, void** input, void** output)
{
    if (codegen(ke) == false)
        return -1.0;
    if (compile(ke) == false)
        return -1.0;
    if (ke->entry_point == nullptr)
        return -1.0;
    return ke->entry_point(input, output);
}

CudaDefaultRuntime::Pointer CudaDefaultRuntime::Runtime()
{
    static CudaDefaultRuntime::Pointer predefined = nullptr;
    if (predefined == nullptr)
        predefined = make_shared<CudaDefaultRuntime>();
    return predefined;
}