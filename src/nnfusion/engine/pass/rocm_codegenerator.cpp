// Microsoft (c) 2019, Wenxiang Hu

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "rocm_codegenerator.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

namespace
{
    std::string generate_cmakelists(void)
    {
        LanguageUnit lu;
        lu << R"(project(main_test)
cmake_minimum_required(VERSION 3.5)

SET(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)

include_directories(
    /opt/rocm/include
    /opt/rocm/rocblas/include
    /opt/rocm/rocrand/include
    /opt/rocm/hiprand/include
    /opt/rocm/hipsparse/include
)

add_library(nnfusion_naive_rt nnfusion_rt.cpp)

add_executable(main_test main_test.cpp)
target_link_libraries(main_test nnfusion_naive_rt MIOpen rocblas)
)";
        return lu.get_code();
    }

    bool create_dir(std::string tar_path)
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
            printf("Directory %s already exists\n", tar_path.c_str());
            flag = true;
        }
        return flag;
    }

    bool save_file(LanguageUnit_p lu)
    {
        std::ofstream out(lu->symbol);
        out << lu->get_code();
        out.close();
        return true;
    }

} // namespace

bool RocmCodeGenerator::setpwd()
{
    std::string tar_path = "./rocm_codegen/";
    create_dir(tar_path);
    int status = chdir(tar_path.c_str());
    return (bool)status;
}

bool RocmCodeGenerator::projgen()
{
    save_file(this->lu_cmakefile);
    save_file(this->lu_nnfusion_rt);
    save_file(this->lu_header);
    save_file(this->lu_main);
}

// bool RocmCodeGenerator::run(std::shared_ptr<InterpreterContext> ctx,
//                         std::shared_ptr<TranslationUnit> tu)
// {
//     auto& p = tu->program;
//     for (auto iterator = p.entry; iterator != nullptr; iterator = iterator->next)
//     {
//         for (auto ins : *iterator)
//         {
//             LOG_INFO << "instruction name: " << ins->name() << ", device: " << ins->Tag().Get<DeviceType>("Device");
//             //ins->Tag().Set<DeviceType>("Device", CUDA_GPU);
//         }
//     }
//     return true;
// }

bool RocmCodeGenerator::run(std::shared_ptr<InterpreterContext> ctx,
                            std::shared_ptr<TranslationUnit> tu)
{
    setpwd();

    this->lu_cmakefile = LanguageUnit_p(new LanguageUnit("CMakeLists.txt"));
    this->lu_nnfusion_rt = LanguageUnit_p(new LanguageUnit("nnfusion_rt.cu"));
    this->lu_header = LanguageUnit_p(new LanguageUnit("nnfusion_rt.h"));
    this->lu_main = LanguageUnit_p(new LanguageUnit("main_test.cpp"));

    LanguageUnit& lu_include = *this->lu_header;

    LanguageUnit lu_kernel_entry("KERNEL_ENTRY"); //func call for main()
    LanguageUnit lu_kernel_entry_header("KERNEL_ENTRY_HEADER");
    LanguageUnit lu_main_init("main_init");         //cuda_init()
    LanguageUnit lu_mem_plan_init("mem_plan_init"); // memory_pool planning in cuda_init()
    LanguageUnit lu_main_free("naive_free");        //cuda_free()

    bool rc = true;

    std::vector<shared_ptr<KernelEmitter>> kernels;
    auto& prog = tu->program;
    for (auto iterator = prog.entry; iterator != nullptr; iterator = iterator->next)
    {
        for (auto ins : *iterator)
        {
            string op_name = ins->operatorDef()->description();

            auto kernel_reg =
                KernelRegistry::Global()->FindKernelRegistration(op_name, ROCM_GPU, DT_FLOAT);
            if (!kernel_reg)
            {
                kernel_reg =
                    KernelRegistry::Global()->FindKernelRegistration(op_name, CUDA_GPU, DT_FLOAT);
            }
            if (!kernel_reg)
            {
                kernel_reg =
                    KernelRegistry::Global()->FindKernelRegistration("AnyOP", CUDA_GPU, DT_FLOAT);
                enforce(kernel_reg != nullptr) << "AnyOp Kernel not found, op=" << op_name;
            }
            shared_ptr<KernelContext> ctx(new KernelContext(ins->operatorDef()));
            auto kernel = kernel_reg->m_factory(ctx);
            kernel->emit_source();
            kernels.push_back(kernel);
        }
    }

    LOG_INFO << "Start dump whole source file...\n";
    // Code Gen
    LanguageUnit& lu = *this->lu_nnfusion_rt;
    lu << "// Microsoft (c) 2019, MSRA/NNFUSION Team\n";

    // Collect Requirement
    unordered_set<string> global_required;
    LanguageUnit re("REQUIREMENT");
    {
        re.require(header::assert);
        re.require(header::stdexcept);
        re.require(header::sstream);
        re.require(macro::CUDA_SAFE_CALL);
        re.require(declaration::typedef_int);

        for (auto kernel : kernels)
        {
            if (!(kernel->is_emitted()))
            {
                return false;
            }

            for (auto& it : kernel->dep_unit->local_symbol)
            {
                re.require(it.second);
                global_required.insert(it.second->symbol);
            }
        }
        // lu << re.collect_required_code();
    }

    lu << "#include \"nnfusion_rt.h\"\n\n";
    lu << "char* _memory_pool;\n\n";

    // Collect Function Definition
    {
        unordered_set<string> declared;
        LanguageUnit def("FUNCTIONS");
        for (auto kernel : kernels)
        {
            for (auto& it : kernel->body_unit->local_symbol)
            {
                if (it.second != kernel->dep_unit)
                {
                    re.require(it.second);
                    global_required.insert(it.second->symbol);
                }
            }
            def << kernel->emit_comments();
            if (declared.count(kernel->body_unit->symbol) == 0)
            {
                def << kernel->signature_unit->get_code() << "\n";
                def.block_begin();
                def << kernel->body_unit->get_code() << "\n";
                def.block_end();
                declared.insert(kernel->body_unit->symbol);
            }
            else
            {
                def << "// Function declared:" << kernel->body_unit->symbol << "\n\n";
            }
        }

        //Write Dependency
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("header::") != string::npos)
                lu << it.second->get_code();
        lu << "\n";
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("macro::") != string::npos)
                lu << it.second->get_code();
        lu << "\n";
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("declaration::") != string::npos)
                lu << it.second->get_code();
        lu << "\n";
        //Write Code
        lu << def.collect_code() << "\n";
    }

    // Generate caller function body
    {
        unordered_set<string> allocated;
        lu << "extern \"C\" int naive_entry(";
        lu_kernel_entry << "extern \"C\" int kernel_entry(";
        // Add param
        {
            vector<string> params;
            for (int i = 0; i < tu->arg.size(); i++)
            {
                auto tv = tu->arg[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << type << "* " << tv->get_name();
                allocated.insert(tv->get_name());
                params.push_back(ss.str());
            }

            for (int i = 0; i < tu->out.size(); i++)
            {
                auto tv = tu->out[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << type << "* " << tv->get_name();
                allocated.insert(tv->get_name());
                params.push_back(ss.str());
            }

            lu << join(params, ", ");
            lu_kernel_entry << join(params, ", ");
        }
        lu << ")\n";
        lu.block_begin();
        lu_kernel_entry << ")";
        lu_kernel_entry_header << lu_kernel_entry.get_code();
        lu_kernel_entry << "\n";
        lu_kernel_entry.block_begin();

        //Planning
        {
            // enforce(tu->memory_pool_size > 0) << "GPU Memory pool size cannot be zero.";
            lu_main_init << "CUDA_SAFE_CALL(cudaMalloc((void**)&_memory_pool, "
                         << tu->memory_pool_size << "));\n";
            lu << "CUDA_SAFE_CALL(cudaMalloc((void**)&_memory_pool, " << tu->memory_pool_size
               << "));\n";

            for (auto kernel : kernels)
            {
                for (auto& it : kernel->m_context->inputs)
                {
                    if (allocated.count(it.get_name()) > 0)
                        continue;
                    lu << it.get_type() << "* " << it.get_name() << " = (" << it.get_type()
                       << "*)(_memory_pool+" << it.get_offset() << ");\n";
                    allocated.insert(it.get_name());

                    lu_mem_plan_init << it.get_type() << "* " << it.get_name() << ";\n";
                    lu_main_init << it.get_name() << " = (" << it.get_type() << "*)(_memory_pool+"
                                 << it.get_offset() << ");\n";
                }

                for (auto& it : kernel->m_context->outputs)
                {
                    if (allocated.count(it.get_name()) > 0)
                        continue;
                    lu << it.get_type() << "* " << it.get_name() << " = (" << it.get_type()
                       << "*)(_memory_pool+" << it.get_offset() << ");\n";
                    allocated.insert(it.get_name());

                    lu_mem_plan_init << it.get_type() << "* " << it.get_name() << ";\n";
                    lu_main_init << it.get_name() << " = (" << it.get_type() << "*)(_memory_pool+"
                                 << it.get_offset() << ");\n";
                }
            }
        }

        //Function Call
        {
            if (global_required.count("declaration::global_cublas_handle") > 0)
            {
                lu << "CUBLAS_SAFE_CALL(cublasCreate(&global_cublas_handle));\n";
                lu_main_init << "CUBLAS_SAFE_CALL(cublasCreate(&global_cublas_handle));\n";
            }
            if (global_required.count("declaration::global_cudnn_handle") > 0)
            {
                lu << "CUDNN_SAFE_CALL(miopenCreate(&global_cudnn_handle));\n";
                lu_main_init << "CUDNN_SAFE_CALL(miopenCreate(&global_cudnn_handle));\n";
            }
            if (global_required.count("declaration::num_SMs") > 0)
            {
                lu << "CUDA_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, "
                      "cudaDevAttrMultiProcessorCount, 0));\n";
                lu_main_init << "CUDA_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, "
                                "cudaDevAttrMultiProcessorCount, 0));\n";
            }
            for (auto kernel : kernels)
            {
                lu << kernel->call_unit->get_code();
                // lu << "assert(cudaSuccess == cudaGetLastError());\n";
                std::string read_const = kernel->call_unit->get_code();
                if (read_const.compare(0, 10, "read_const") == 0)
                {
                    lu_main_init << kernel->call_unit->get_code();
                }
                else
                {
                    lu_kernel_entry << kernel->call_unit->get_code();
                }
            }
            if (global_required.count("declaration::global_cublas_handle") > 0)
            {
                lu << "CUBLAS_SAFE_CALL(rocblas_create_handle(&global_cublas_handle));\n";
                lu_main_free << "CUBLAS_SAFE_CALL(rocblas_destroy_handle(global_cublas_handle));\n";
            }
            if (global_required.count("declaration::global_cudnn_handle") > 0)
            {
                lu << "CUDNN_SAFE_CALL(miopenDestroy(global_cudnn_handle));\n";
                lu_main_free << "CUDNN_SAFE_CALL(miopenDestroy(global_cudnn_handle));\n";
            }
        }

        lu << "CUDA_SAFE_CALL(cudaFree(_memory_pool));\n";
        lu << "return 0;\n";

        lu_main_free << "CUDA_SAFE_CALL(cudaFree(_memory_pool));\n";
        lu.block_end();

        lu_kernel_entry << "return 0;\n";
        lu_kernel_entry.block_end();
    }

    lu << "\n";
    {
        lu << lu_mem_plan_init.get_code();
        lu << "\nextern \"C\" void cuda_init()";
        lu.block_begin();
        {
            lu << lu_main_init.get_code();
        }
        lu.block_end();

        lu << "extern \"C\" void cuda_free()";
        lu.block_begin();
        {
            lu << lu_main_free.get_code();
        }

        lu.block_end();
    }
    lu << "\n";
    lu << lu_kernel_entry.get_code() << "\n\n";

    // Test function
    {
        lu << "extern \"C\" int naive_test(";
        // Add param
        {
            vector<string> params;
            for (int i = 0; i < tu->arg.size(); i++)
            {
                auto tv = tu->arg[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << type << "* " << tv->get_name() << "_host";
                params.push_back(ss.str());
            }

            for (int i = 0; i < tu->out.size(); i++)
            {
                auto tv = tu->out[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << type << "* " << tv->get_name() << "_host";
                params.push_back(ss.str());
            }

            lu << join(params, ", ");
        }
        lu << ")\n";
        lu.block_begin();
        {
            for (size_t i = 0; i < tu->arg.size(); i++)
            {
                auto& tensor = *tu->arg[i];
                lu << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                   << ";\n"
                   << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ","
                   << tensor.get_tensor_layout()->get_size() << " * "
                   << tensor.get_element_type().size() << "));\n";

                lu << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << ", " << tensor.get_name()
                   << "_host, " << tensor.get_tensor_layout()->get_size() << " * "
                   << tensor.get_element_type().size() << ", "
                   << "cudaMemcpyHostToDevice));\n";
            }

            lu << "//output arguments\n";
            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                lu << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                   << ";\n"
                   << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ","
                   << tensor.get_tensor_layout()->get_size() << " * "
                   << tensor.get_element_type().size() << "));\n";
            }

            vector<string> params;
            for (int i = 0; i < tu->arg.size(); i++)
            {
                auto& tv = tu->arg[i];
                params.push_back(tv->get_name());
            }

            for (int i = 0; i < tu->out.size(); i++)
            {
                auto& tv = tu->out[i];
                params.push_back(tv->get_name());
            }

            lu << "naive_entry(" << join(params, ", ") << ");\n";

            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                lu << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << "_host, "
                   << tensor.get_name() << ", " << tensor.get_tensor_layout()->get_size() << " * "
                   << tensor.get_element_type().size() << ", "
                   << "cudaMemcpyDeviceToHost));\n";
            }

            for (size_t i = 0; i < tu->arg.size(); i++)
            {
                auto& tensor = *tu->arg[i];
                lu << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
            }

            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                lu << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
            }
        }
        lu << "return 0;\n";
        lu.block_end();
    }

    lu << "\n";

    // Test function 2
    {
        lu << "extern \"C\" int naive_test_simple(void** args)\n";
        // Add param
        lu.block_begin();
        {
            lu << "return naive_test(";
            vector<string> params;
            int acc = 0;
            for (int i = 0; i < tu->arg.size(); i++, acc++)
            {
                auto tv = tu->arg[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << "(" << type << "*)args[" << acc << "]";
                params.push_back(ss.str());
            }

            for (int i = 0; i < tu->out.size(); i++, acc++)
            {
                auto tv = tu->out[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << "(" << type << "*)args[" << acc << "]";
                params.push_back(ss.str());
            }
            lu << join(params, ", ");
            lu << ");\n";
        }
        lu.block_end();
    }

    // generate main() function
    std::string function_include =
        "#include \"nnfusion_rt.h\"\n#include <stdlib.h>\n#include <stdio.h>\n";
    LanguageUnit& lu_main = *this->lu_main;
    {
        lu_main << function_include << "\n";
        lu_main << header::stdexcept->get_code();
        lu_main << "#include <sstream>\n";
        lu_main << macro::CUDA_SAFE_CALL->get_code();
        lu_main << "\n";
        lu_main << "int main(void)";
        lu_main.block_begin();
        {
            lu_main << "\ncuda_init();\n\n";

            for (size_t i = 0; i < tu->arg.size(); i++)
            {
                auto& tensor = *tu->arg[i];
                //malloc host input arg
                lu_main << "//input argument\n";
                lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                        << "_host = (" << tensor.get_element_type().c_type_string() << "*)"
                        << "malloc( sizeof(" << tensor.get_element_type().c_type_string() << ")* "
                        << tensor.get_tensor_layout()->get_size() << " * "
                        << tensor.get_element_type().size() << ");\n\n";

                //cudaMalloc input arg
                lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                        << ";\n"
                        << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ","
                        << tensor.get_tensor_layout()->get_size() << " * "
                        << tensor.get_element_type().size() << "));\n";

                lu_main << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << ", "
                        << tensor.get_name() << "_host, " << tensor.get_tensor_layout()->get_size()
                        << " * " << tensor.get_element_type().size() << ", "
                        << "cudaMemcpyHostToDevice));\n";
            }

            lu_main << "\n//output arguments\n";
            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                //malloc host output arg
                lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                        << "_host = (" << tensor.get_element_type().c_type_string() << "*)"
                        << "malloc( sizeof(" << tensor.get_element_type().c_type_string() << ")* "
                        << tensor.get_tensor_layout()->get_size() << " * "
                        << tensor.get_element_type().size() << ");\n\n";

                //cudaMalloc output args
                lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                        << ";\n"
                        << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ","
                        << tensor.get_tensor_layout()->get_size() << " * "
                        << tensor.get_element_type().size() << "));\n";
            }

            vector<string> params;
            for (int i = 0; i < tu->arg.size(); i++)
            {
                auto& tv = tu->arg[i];
                params.push_back(tv->get_name());
            }
            for (int i = 0; i < tu->out.size(); i++)
            {
                auto& tv = tu->out[i];
                params.push_back(tv->get_name());
            }

            lu_main << "\n//GPU time measurement\n";
            lu_main << "cudaEvent_t start, stop;\n";
            lu_main << "cudaEventCreate(&start);\n";
            lu_main << "cudaEventCreate(&stop);\n";

            lu_main << "\n//time measurement\n";
            lu_main << "cudaEventRecord(start);\n\n";
            lu_main << "//kernel call\n";

            lu_main << "int steps = 100;\n";
            lu_main << "for(int i_=0; i_<steps; i_++)\n";
            lu_main.block_begin();

            // kernel launch
            lu_main << "kernel_entry(" << join(params, ", ") << ");\n";

            lu_main.block_end();

            lu_main << "//time measurement\n";
            lu_main << "\ncudaEventRecord(stop);\n";
            lu_main << "cudaEventSynchronize(stop);\n";
            lu_main << "float milliseconds = 0;\n";
            lu_main << "cudaEventElapsedTime(&milliseconds, start, stop);\n";
            lu_main << "printf(\"function execution time: %f ms\\n\", milliseconds/steps);\n";
            lu_main << "\n//free context\n";

            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                lu_main << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << "_host, "
                        << tensor.get_name() << ", " << tensor.get_tensor_layout()->get_size()
                        << " * " << tensor.get_element_type().size() << ", "
                        << "cudaMemcpyDeviceToHost));\n";
            }

            for (size_t i = 0; i < tu->arg.size(); i++)
            {
                auto& tensor = *tu->arg[i];
                lu_main << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
            }

            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                lu_main << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
            }
            lu_main << "cuda_free();\n\n";
        }
        //free host input args
        for (size_t i = 0; i < tu->arg.size(); i++)
        {
            auto& tensor = *tu->arg[i];
            lu_main << "free(" << tensor.get_name() << "_host);\n";
        }
        //free host output args
        for (size_t i = 0; i < tu->out.size(); i++)
        {
            auto& tensor = *tu->out[i];
            lu_main << "free(" << tensor.get_name() << "_host);\n";
        }

        lu_main << "return 0;\n";
        lu_main.block_end();
    }

    //generate include header file
    lu_include << "// Microsoft (c) 2019\n";
    lu_include << "#pragma once\n";
    lu_include << "extern \"C\" int naive_test_simple(void** args);\n";
    lu_include << lu_kernel_entry_header.get_code() << ";\n";
    lu_include << "extern \"C\" void cuda_init();\n";
    lu_include << "extern \"C\" void cuda_free();\n";
    lu_include << header::cuda->get_code();

    //generate CMakeList.txt
    LanguageUnit& lu_cmake = *this->lu_cmakefile;
    lu_cmake << generate_cmakelists();

    projgen();

    // hipify kernel codes
    char exepath[1024];
    assert(readlink("/proc/self/exe", exepath, sizeof(exepath)) > 0);
    for (int i = strlen(exepath) - 1; i >= 0; --i)
        if (exepath[i] == '/')
        {
            exepath[i] = 0;
            break;
        }
    assert(0 == system((std::string(exepath) +
                        "/hipify-nnfusion nnfusion_rt.cu | grep -v 'include.*cublas_v2' | grep -v "
                        "'include.*cuda.h' > nnfusion_rt.cpp && rm nnfusion_rt.cu")
                           .c_str()));
    assert(0 == system("sed -i 's/<cuda\\.h>/\"rocm_adapter.h\"/g' nnfusion_rt.h && sed -i "
                       "'s/cuda_runtime\\.h/hip\\/hip_runtime.h/g' nnfusion_rt.h"));
    assert(0 ==
           system((std::string("cp ") + exepath + "/hipify-adapter ./rocm_adapter.h").c_str()));
    return rc;
}
