// Microsoft (c) 2019, Wenxiang Hu

#include <libgen.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "codegenerator_helper.hpp"
#include "cuda_codegenerator.hpp"
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/core/kernels/cpu/barrier.hpp"
#include "nnfusion/core/kernels/cpu/cpu_langunit.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/engine/memory_allocator.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::async;

DEFINE_bool(fcodegen_debug, false, "Add debug functions in Codegen-ed project.");
DEFINE_bool(fcodegen_timing, false, "Add timing functions in Codegen-ed project.");
DECLARE_bool(frt_const_folding);
DECLARE_bool(fkernels_as_files);
DECLARE_int64(fkernels_files_number);
//DECLARE_int32(fnuma_node_num);
//DECLARE_int32(fthread_num_per_node);

namespace
{
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

// todo: add flags for future.
std::string CudaCodeGenerator::get_generate_cmakelists()
{
    LanguageUnit lu;
    lu << R"(project(main_test)
cmake_minimum_required(VERSION 3.5)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -cudart shared")

link_directories(/usr/local/cuda/lib64)

find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)
)" << (super_scaler_enable ? "find_package(MPI)" : "")
       << R"(
include_directories(${CUDNN_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} )"
       << (super_scaler_enable ? "${MPI_INCLUDE_PATH}" : "") << R"()

find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

find_library(CUDA_cuda_LIBRARY cuda /usr/local/cuda/lib64/stubs)
find_library(CUDA_cudart_LIBRARY libcudart.so /usr/local/cuda/lib64)
)" << (super_scaler_enable ? "find_library(SUPER_SCALER_LIBRARIES libsuper_scaler.so "
                             "${CMAKE_CURRENT_SOURCE_DIR})"
                           : "")
       << (FLAGS_fkernels_as_files
               ? "file(GLOB kernels kernels/*.cu)\ncuda_add_library(nnfusion_naive_rt "
                 "nnfusion_rt.cu ${kernels})\n"
               : "cuda_add_library(nnfusion_naive_rt nnfusion_rt.cu)\n")
       << R"(

target_link_libraries(nnfusion_naive_rt
    ${CUDA_cuda_LIBRARY}
    ${CUDA_cudart_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDNN_LIBRARIES})"
       << (super_scaler_enable ? R"(
    ${MPI_LIBRARIES}
    ${SUPER_SCALER_LIBRARIES}
    nccl)"
                               : "")
       << R"(
)

cuda_add_executable(main_test main_test.cpp)
target_link_libraries(main_test nnfusion_naive_rt cudnn culibos cublas)
if(EXISTS "${CMAKE_BINARY_DIR}/Constant")
else()
add_custom_command(
    TARGET nnfusion_naive_rt
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/Constant ${CMAKE_BINARY_DIR}/Constant
)
endif()
)";
    return lu.get_code();
}

void CudaCodeGenerator::post_projgen()
{
    // create_image_tests
    nnfusion::codegen::copy_file_from_templates("image_tests/image_test.cpp",
                                                "./image_tests/image_test.cpp");
    nnfusion::codegen::copy_file_from_templates("image_tests/CMakeLists_cuda.txt",
                                                "./image_tests/CMakeLists.txt");
}

std::string CudaCodeGenerator::get_target_name()
{
    return "cuda_codegen";
}

bool CudaCodeGenerator::setpwd(std::shared_ptr<InterpreterContext> ctx,
                               std::shared_ptr<TranslationUnit> tu)
{
    std::string working_dir = "./nnfusion_rt";
    create_dir(working_dir);
    std::string tar_path = working_dir + "/" + get_target_name() + "/";
    if (ctx->m_graphs.size() > 1)
    {
        create_dir(tar_path);
        tar_path += tu->m_graph->get_name() + "/";
    }
    std::string kernels_path = tar_path + "kernels/";
    create_dir(tar_path);
    if (FLAGS_fkernels_as_files)
        create_dir(kernels_path);
    int status = chdir(tar_path.c_str());
    return (bool)status;
}

bool CudaCodeGenerator::projgen()
{
    save_file(this->lu_cmakefile);
    save_file(this->lu_nnfusion_rt);
    save_file(this->lu_header);
    save_file(this->lu_main);
    return true;
}

void CudaCodeGenerator::after_projgen()
{
    if (super_scaler_enable)
    {
        nnfusion::codegen::copy_file_from_templates("super_scaler/super_scaler.h",
                                                    "./super_scaler.h");
        NNFUSION_LOG(NNFUSION_WARNING) << "libsuper_scaler.so should be copied from "
                                          "(build)/src/tools/nnfusion/templates/super_scaler/";
        nnfusion::codegen::copy_file_from_templates("super_scaler/libsuper_scaler.so",
                                                    "./libsuper_scaler.so");
    }
}

// bool CudaCodeGenerator::run(std::shared_ptr<InterpreterContext> ctx,
//                         std::shared_ptr<TranslationUnit> tu)
// {
//     auto& p = tu->program;
//     for (auto iterator = p.entry; iterator != nullptr; iterator = iterator->next)
//     {
//         for (auto ins : *iterator)
//         {
//             NNFUSION_LOG(INFO) << "instruction name: " << ins->name() << ", device: " << ins->Tag().Get<NNFusion_DeviceType>("Device");
//             //ins->Tag().Set<NNFusion_DeviceType>("Device", CUDA_GPU);
//         }
//     }
//     return true;
// }

std::vector<shared_ptr<const KernelRegistration>>
    CudaCodeGenerator::find_backend_kernels(const std::string& op_name,
                                            const shared_ptr<KernelContext>& ctx)
{
    return KernelRegistry::Global()->FindKernelRegistrations(op_name, CUDA_GPU, DT_FLOAT);
}

// KernelEmitter::Pointer
//     CudaCodeGenerator::match_kernel(std::vector<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>& res)
// {
//     for (auto& k : res)
//     {
//         if (k.second != nullptr && k.first == device_type() &&
//             k.second->get_or_emit_source() != nullptr)
//         {
//             return k.second;
//         }
//     }
//     return nullptr;
// }

nnfusion::LanguageUnit_p CudaCodeGenerator::func_call_codegen(
    nnfusion::ir::Instruction::Pointer ins, bool func_call_only, const std::string& func_call)
{
    auto CUDA_async_manager = AsyncManagerFactory::get_async_manager(m_graph, CUDA_GPU);
    auto CPU_async_manager = AsyncManagerFactory::get_async_manager(m_graph, GENERIC_CPU);
    auto& async_info = (*ins)["Async_info"].as<AsyncExecutionInfo>();
    LanguageUnit_p _lu(new LanguageUnit("func_call"));
    auto& lu = *_lu;
    if (!func_call_only)
    {
        if (!async_info.wait_barriers.empty())
        {
            for (auto barrier : async_info.wait_barriers)
            {
                lu << CPU_async_manager->emit_event_wait(async_info.execution_thread, barrier)
                          ->get_code();
            }
        }

        if (!async_info.wait_events.empty())
        {
            for (auto event : async_info.wait_events)
            {
                lu << CUDA_async_manager->emit_event_wait(async_info.execution_stream, event)
                          ->get_code();
            }
        }
    }
    if (ins->name() == "Memcpy")
    {
        string stream_name = async_info.execution_stream->get_name();
        auto& inputs = ins->get_inputs();
        NNFUSION_CHECK(inputs.size() == 1);
        auto src_tensor = inputs[0];

        auto& outputs = ins->get_outputs();
        NNFUSION_CHECK(outputs.size() == 1);
        auto dst_tensor = outputs[0];
        NNFUSION_CHECK(src_tensor->get_device_type() == CUDA_GPU ||
                       src_tensor->get_device_type() == ROCM_GPU);
        NNFUSION_CHECK(dst_tensor->get_device_type() == CUDA_GPU ||
                       dst_tensor->get_device_type() == ROCM_GPU);

        lu << "cudaMemcpyAsync(" << dst_tensor->get_name() << ", " << src_tensor->get_name() << ", "
           << dst_tensor->size() << ", cudaMemcpyDeviceToDevice, " << stream_name << ");\n";
    }
    else
    {
        if (ins->getKernel()->is_eliminative())
        {
            lu << "// eliminated\n";
        }
        else
        {
            lu << func_call;
        }
    }
    if (!func_call_only)
    {
        if (async_info.record_event != nullptr)
        {
            lu << CUDA_async_manager->emit_event_record(async_info.record_event)->get_code();
        }
        if (async_info.notify_barrier != nullptr)
        {
            lu << CPU_async_manager->emit_event_record(async_info.notify_barrier)->get_code();
        }
    }

    return _lu;
}

bool CudaCodeGenerator::run(std::shared_ptr<InterpreterContext> ctx,
                            std::shared_ptr<TranslationUnit> tu)
{
    setpwd(ctx, tu);
    m_graph = tu->m_graph;
    NNFUSION_CHECK_NOT_NULLPTR(tu->memory_allocator_factory);
    auto& allocator_list = tu->memory_allocator_factory->get_allocator_list();
    auto CUDA_async_manager = AsyncManagerFactory::get_async_manager(m_graph, CUDA_GPU);
    auto CPU_async_manager = AsyncManagerFactory::get_async_manager(m_graph, GENERIC_CPU);

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
    LanguageUnit lu_thread_func_call("THREAD_FUNCTION_CALL");

    bool rc = true;
    auto& prog = tu->program;
    for (auto iterator : prog)
    {
        for (auto ins : *iterator)
        {
            if (ins->name() == "Memcpy" || (ins->getGNode() && ins->getGNode()->is_parameter()))
            {
                continue;
            }
            // emit constant kernel
            if (ins->getGNode() && ins->getGNode()->is_constant())
            {
                auto kernel = (*ins)["Kernel_Selection_Result"]
                                  .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>()
                                  .second;
                kernel->get_or_emit_source();
                continue;
            }
            if (!ins->getKernel())
            {
                auto kernel_reg =
                    KernelRegistry::Global()->FindKernelRegistration("AnyOP", CUDA_GPU, DT_FLOAT);
                NNFUSION_CHECK(kernel_reg != nullptr) << "AnyOp Kernel not found, op="
                                                      << ins->getGNode()->get_op_type();
                shared_ptr<KernelContext> ctx(new KernelContext(ins->getGNode()));
                auto kernel = kernel_reg->m_factory(ctx);
                kernel->get_or_emit_source();
                ins->setKernel(kernel);
            }
        }
    }
    NNFUSION_LOG(INFO) << "Start dump whole source file...\n";
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
        re.require(header::fstream);
        re.require(header::thread);
        // Both intra_node parallelism and multi-stream need worker_thread_pool.
        if (CPU_async_manager->num_non_default_stream() > 0)
        {
            re.require(header::threadpool);
        }
        if (CPU_async_manager->num_non_default_stream() > 0)
            re.require(declaration::schedule_thread_pool);
        if (CPU_async_manager->num_event() > 0 || CPU_async_manager->num_non_default_stream() > 0)
            re.require(header::barrier);
        //re.require(declaration::typedef_int);

        for (auto iterator : prog)
        {
            for (auto ins : *iterator)
            {
                if (ins->name() == "Memcpy" || (ins->getGNode() && ins->getGNode()->is_parameter()))
                {
                    continue;
                }
                auto kernel = ins->getKernel();
                if (!kernel->is_emitted())
                    return false;
                for (auto& it : kernel->get_or_emit_source()->dep_unit->local_symbol)
                {
                    re.require(it.second);
                    global_required.insert(it.second->symbol);
                }
            }
        }
    }

    lu << "#include \"nnfusion_rt.h\"\n\n";
    unordered_map<string, LanguageUnit_p> decleard_function_LU;

    // Collect Function Definition
    {
        vector<codegenerator::FunctionFile> cuda_kernel_files;
        if (FLAGS_fkernels_as_files && FLAGS_fkernels_files_number > 0)
            cuda_kernel_files.resize(FLAGS_fkernels_files_number);
        int cuda_kernel_n = 0;
        LanguageUnit def("FUNCTIONS");
        int count_k = 0;
        for (auto iterator : prog)
        {
            for (auto ins : *iterator)
            {
                auto kernel = ins->getKernel();
                auto gnode = ins->getGNode();
                if (ins->name() == "Memcpy" || (gnode && gnode->is_parameter()) ||
                    (kernel && kernel->is_eliminative()))
                    continue;
                //get kernel's stream name
                auto& async_info = (*ins)["Async_info"].as<AsyncExecutionInfo>();
                int device_id = (*ins)["DeviceID"].as<int>();
                string stream_name = async_info.execution_stream->get_name();

                FunctionUnit_p fu = kernel->get_or_emit_source();
                for (auto& it : fu->body_unit->local_symbol)
                {
                    if (it.second != fu->dep_unit)
                    {
                        re.require(it.second);
                        global_required.insert(it.second->symbol);
                    }
                }
                string body_unit = fu->body_unit->get_code();

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
                string func_key = fu->signature_unit->get_code() + body_unit;
                if (kernel->is_static_function() ||
                    decleard_function_LU.find(func_key) == decleard_function_LU.end())
                {
                    auto functionfile = codegenerator::FunctionFile::convert_from(kernel);
                    if (FLAGS_fkernels_as_files)
                    {
                        def << functionfile->get_extern_declare();
                        if (FLAGS_fkernels_files_number > 0)
                            cuda_kernel_files[cuda_kernel_n].merge_from(functionfile);
                        else
                            functionfile->save_file();
                    }
                    else
                    {
                        def << functionfile->get_code();
                    }
                    if (!kernel->is_static_function())
                    {
                        decleard_function_LU[func_key] = fu->name_unit;
                    }
                }
                else
                {
                    //def << "// Function declared:" << kernel->body_unit->symbol << "\n\n";
                }
                if (FLAGS_fkernels_files_number > 0)
                {
                    cuda_kernel_n++;
                    cuda_kernel_n %= FLAGS_fkernels_files_number;
                }
            }
        }
        if (FLAGS_fkernels_as_files && FLAGS_fkernels_files_number > 0)
        {
            for (int i = 0; i < FLAGS_fkernels_files_number; i++)
                cuda_kernel_files[i].save_file();
        }

        //Write Dependency
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("header::") != string::npos)
                lu << it.second->get_code();
        lu << "#include <cstring>\n";
        // lu << "using namespace std;\n";
        lu << "\n";
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("macro::") != string::npos)
                lu << it.second->get_code();
        lu << "\n";
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("declaration::") != string::npos)
            {
                // This for dealing with Concat Op's special cases;
                if (FLAGS_fkernels_as_files &&
                    it.second->symbol.find("_private_kernels") != string::npos)
                    continue;
                lu << it.second->get_code();
            }
        lu << "\n";

        // stream and event declaration
        if (CUDA_async_manager->num_stream() > 0)
            lu << CUDA_async_manager->emit_stream_decl()->get_code();
        if (CUDA_async_manager->num_event() > 0)
            lu << CUDA_async_manager->emit_event_decl()->get_code();
        // thread and barrier declaration
        if (CPU_async_manager->num_stream() > 0)
            lu << CPU_async_manager->emit_stream_decl()->get_code();
        if (CPU_async_manager->num_event() > 0)
            lu << CPU_async_manager->emit_event_decl()->get_code();
        // default barrier declaration
        if (CPU_async_manager->num_non_default_stream() > 0)
            lu << "nnfusion::cpu::Barrier default_barrier("
               << CPU_async_manager->num_non_default_stream() << ");\n";
        //Write Code
        lu << def.collect_code() << "\n";
        if (CPU_async_manager->num_event() > 0 || CPU_async_manager->num_non_default_stream() > 0)
            save_file(barrier_header);
    }

    bool enable_debug = FLAGS_fcodegen_debug;
    bool enable_timing = FLAGS_fcodegen_timing;
    bool enable_rt_const_folding = FLAGS_frt_const_folding;

    // Generate caller function body
    {
        //Planning
        size_t total_alloc = 0;
        for (const auto& allocator : allocator_list)
        {
            total_alloc += allocator.second->max_allocated();
        }
        lu_main_init << "// total memory: " << total_alloc << "\n";
        for (const auto& allocator : allocator_list)
        {
            lu_mem_plan_init << allocator.second->emit_memory_init()->get_code();
            lu_main_init << allocator.second->emit_memory_alloc()->get_code();
        }
        //Function Call
        {
            if (CPU_async_manager->num_non_default_stream() > 0)
            {
                lu_main_init << "schedule_thread_pool = new concurrency::NumaAwareThreadPool();\n";
            }

            if (CUDA_async_manager->num_stream() > 0)
            {
                lu_main_init << "// create streams/handles\n";
                lu_main_init << CUDA_async_manager->emit_stream_init()->get_code();
            }
            if (CUDA_async_manager->num_event() > 0)
            {
                lu_main_init << " // create events\n";
                lu_main_init << CUDA_async_manager->emit_event_init()->get_code();
            }
            if (global_required.count("declaration::num_SMs") > 0)
            {
                lu_main_init << "CUDA_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, "
                                "cudaDevAttrMultiProcessorCount, 0));\n";
            }
            //const
            set<string> constant_vals;
            std::unordered_map<string, vector<nnfusion::ir::Instruction::Pointer>>
                thread_kernels_entry;
            std::unordered_map<shared_ptr<KernelEmitter>, string> kernel_func_call;
            int pre_dev_id = 0;
            lu_main_init << " // func call\n";
            lu_main_init << "CUDA_SAFE_CALL(cudaSetDevice(" << pre_dev_id << "));\n";
            for (auto iterator : prog)
            {
                for (auto ins : *iterator)
                {
                    auto kernel = ins->getKernel();
                    auto gnode = ins->getGNode();
                    auto& async_info = (*ins)["Async_info"].as<AsyncExecutionInfo>();
                    int device_id = (*ins)["DeviceID"].as<int>();
                    auto thread = async_info.execution_thread;
                    auto thread_name = thread->get_name();

                    if (!kernel)
                    {
                        if (ins->name() == "Memcpy")
                        {
                            if ((*ins)["Memcpy_Constant_or_Variable"].is_valid_as<bool>() ||
                                (enable_rt_const_folding &&
                                 (*ins)["rt_const_folding"].is_valid_as<bool>()))
                            {
                                // lu_main_init << "CUDA_SAFE_CALL(cudaSetDevice(" << device_id
                                //              << "));\n";
                                lu_main_init << func_call_codegen(ins, true)->get_code();
                            }
                            else
                            {
                                thread_kernels_entry[thread_name].push_back(ins);
                            }
                        }
                        continue;
                    }
                    FunctionUnit_p fu = kernel->get_or_emit_source(true);
                    std::string func_name;
                    if (!kernel->is_eliminative())
                    {
                        if (kernel->is_static_function())
                        {
                            func_name = fu->name_unit->get_code();
                        }
                        else
                        {
                            std::string body_unit = fu->body_unit->get_code();
                            // conv kernels in the the stream shares the same workspace_ptr
                            if (gnode->get_op_type() == "Convolution")
                            {
                                std::string s_workspace =
                                    "workspace_ptr_" +
                                    to_string(async_info.execution_stream->get_stream_id());
                                int pos = body_unit.find("workspace_ptr");
                                while (pos >= 0)
                                {
                                    body_unit.replace(pos, 13, s_workspace);
                                    pos = body_unit.find("workspace_ptr", pos + s_workspace.size());
                                }
                            }
                            string func_key = fu->signature_unit->get_code() + body_unit;
                            NNFUSION_CHECK(decleard_function_LU.find(func_key) !=
                                           decleard_function_LU.end());
                            func_name = decleard_function_LU[func_key]->get_code();
                        }
                    }
                    // get kernel func call
                    std::string function_call = fu->get_specialized_funciton_call(func_name);
                    int pos_right = function_call.find(">>>(");
                    if (pos_right >= 0)
                    {
#ifdef __USING_HOST_CALL_FORMAT___
                        // Turn to Host Call Format in kernel_entry()
                        int pos_left = function_call.find("<<<");
                        NNFUSION_CHECK(pos_left >= 0);
                        function_call = function_call.substr(0, pos_left) + "_Call(" +
                                        function_call.substr(pos_left + sizeof("<<<") - 1);

                        pos_right = function_call.find(">>>(");
                        NNFUSION_CHECK(pos_right >= 0);
                        function_call = function_call.substr(0, pos_right) + ", " +
                                        function_call.substr(pos_right + sizeof(">>>(") - 1);
#endif
                    }
                    kernel_func_call[kernel] = function_call;

                    if (gnode->is_constant() || gnode->is_variable() ||
                        (enable_rt_const_folding && (*ins)["rt_const_folding"].is_valid_as<bool>()))
                    {
                        NNFUSION_CHECK(async_info.execution_stream->is_default_stream())
                            << "Kernel function calls in cuda_init() should use default stream.";
                        if (device_id != pre_dev_id)
                        {
                            lu_main_init << "CUDA_SAFE_CALL(cudaSetDevice(" << device_id << "));\n";
                            pre_dev_id = device_id;
                        }
                        auto function_call = kernel_func_call[kernel];
                        lu_main_init << func_call_codegen(ins, true, function_call)->get_code();
                    }
                    else
                    {
                        thread_kernels_entry[thread->get_name()].push_back(ins);
                    }
                }
            }

            // Generate graph configs
            {
                lu_kernel_entry << "\n#ifndef __NNFUSION_GRAPH_CONFIG__\n";
                lu_kernel_entry << "#define __NNFUSION_GRAPH_CONFIG__\n";
                lu_kernel_entry << "#define NNFUSION_GRAPH_INPUT_NUM " << tu->arg.size() << "\n";
                lu_kernel_entry << "#define NNFUSION_GRAPH_OUTPUT_NUM " << tu->out.size() << "\n";
                for (int i = 0; i < tu->arg.size(); i++)
                {
                    lu_kernel_entry << "#define NNFUSION_GRAPH_INPUT_DTYPE_" << i << " "
                                    << tu->arg[i]->get_element_type().c_type_string() << "\n";
                    lu_kernel_entry << "#define NNFUSION_GRAPH_INPUT_SHAPE_" << i << " {";
                    auto& shape = tu->arg[i]->get_shape();
                    for (int j = 0; j < shape.size(); ++j)
                    {
                        if (j > 0)
                            lu_kernel_entry << ", ";
                        lu_kernel_entry << shape[j];
                    }
                    lu_kernel_entry << "}\n";
                }
                for (int i = 0; i < tu->out.size(); i++)
                {
                    lu_kernel_entry << "#define NNFUSION_GRAPH_OUTPUT_DTYPE_" << i << " "
                                    << tu->out[i]->get_element_type().c_type_string() << "\n";
                    lu_kernel_entry << "#define NNFUSION_GRAPH_OUTPUT_SHAPE_" << i << " {";
                    auto& shape = tu->out[i]->get_shape();
                    for (int j = 0; j < shape.size(); ++j)
                    {
                        if (j > 0)
                            lu_kernel_entry << ", ";
                        lu_kernel_entry << shape[j];
                    }
                    lu_kernel_entry << "}\n";
                }
                lu_kernel_entry << "#endif\n\n";
            }

            // emit function calls in kernel_entry()
            unordered_set<string> allocated;
            std::string kernel_entry_params;
            std::string kernel_entry_args;

            lu_kernel_entry << "extern \"C\" int kernel_entry(";
            // Add param
            {
                vector<string> params;
                vector<string> args;
                for (int i = 0; i < tu->arg.size(); i++)
                {
                    auto tv = tu->arg[i];
                    string type = tv->get_element_type().c_type_string();
                    stringstream ss;
                    ss << type << "* " << tv->get_name();
                    allocated.insert(tv->get_name());
                    params.push_back(ss.str());
                    args.push_back(tv->get_name());
                }

                for (int i = 0; i < tu->out.size(); i++)
                {
                    auto tv = tu->out[i];
                    string type = tv->get_element_type().c_type_string();
                    stringstream ss;
                    ss << type << "** " << tv->get_name();
                    allocated.insert(tv->get_name());
                    params.push_back(ss.str());
                    args.push_back(tv->get_name());
                }
                kernel_entry_params = join(params, ", ");
                kernel_entry_args = join(args, ", ");

                lu_kernel_entry << join(params, ", ");
            }
            lu_kernel_entry << ")";
            lu_kernel_entry_header << lu_kernel_entry.get_code();
            lu_kernel_entry << "\n";
            lu_kernel_entry.block_begin();

            // reset event/notification
            lu_kernel_entry << CPU_async_manager->emit_event_reset()->get_code();
            if (CPU_async_manager->num_non_default_stream() > 0)
                lu_kernel_entry << "default_barrier.Reset();\n";

            int thread_index = 0;
            int thread_func_call_count = 1;
            for (auto& tk : thread_kernels_entry)
            {
                size_t kernel_order = 0;
                auto& thread_name = tk.first;
                NNFUSION_CHECK(tk.second.size() > 0);
                int device_id = (*tk.second[0])["DeviceID"].as<int>();
                if (thread_name != "default_thread")
                {
                    // add thread_calls definition
                    lu_thread_func_call << "extern \"C\" void " << thread_name << "(";
                    lu_thread_func_call << kernel_entry_params << ")\n";
                    lu_thread_func_call.block_begin();
                    lu_thread_func_call << "CUDA_SAFE_CALL(cudaSetDevice(" << device_id << "));\n";
                    if (enable_timing)
                    {
                        lu_thread_func_call << "static cudaEvent_t hEvents[" << tk.second.size() + 1
                                            << "];\n";
                        lu_thread_func_call << "size_t eventCnt = 0;\n";
                        lu_thread_func_call
                            << "if (!hEvents[eventCnt]) "
                               "CUDA_SAFE_CALL(cudaEventCreate(&hEvents[eventCnt]));\n";
                        lu_thread_func_call
                            << "CUDA_SAFE_CALL(cudaEventRecord(hEvents[eventCnt++]));\n";
                    }

                    for (auto ins : tk.second)
                    {
                        auto kernel = ins->getKernel();
                        if (!kernel)
                        {
                            if (ins->name() == "Memcpy")
                            {
                                lu_thread_func_call << " // order=" << ++kernel_order
                                                    << ", name=memcpy\n";
                                lu_thread_func_call << func_call_codegen(ins)->get_code();
                                if (enable_timing)
                                {
                                    lu_thread_func_call
                                        << "if (!hEvents[eventCnt]) "
                                           "CUDA_SAFE_CALL(cudaEventCreate(&hEvents[eventCnt]));\n";
                                    lu_thread_func_call << "CUDA_SAFE_CALL(cudaEventRecord(hEvents["
                                                           "eventCnt++]));\n";
                                }
                            }
                            continue;
                        }
                        auto gnode = ins->getGNode();
                        auto& async_info = (*ins)["Async_info"].as<AsyncExecutionInfo>();
                        const string node_name = gnode ? gnode->get_name() : "internal_node";

                        lu_thread_func_call << " // order=" << ++kernel_order
                                            << ", name=" << node_name << "\n";
                        auto function_call = kernel_func_call[kernel];

                        lu_thread_func_call
                            << func_call_codegen(ins, false, function_call)->get_code();

                        if (enable_debug && !gnode->get_op_ptr()->is_output())
                        {
                            for (size_t i = 0; i < kernel->m_context->outputs.size(); i++)
                            {
                                if (kernel->m_context->outputs[i]
                                        ->get_element_type()
                                        .c_type_string() != "float")
                                    continue;
                                auto out_name = kernel->m_context->output_names[i];
                                lu_thread_func_call
                                    << "Debug(\"" << node_name << ", " << out_name << "\", "
                                    << out_name << ", \"" << join(kernel->m_context->input_names)
                                    << "\", " << kernel->m_context->outputs[i]->size(false)
                                    << ");\n";
                            }
                        }
                        if (enable_timing)
                        {
                            lu_thread_func_call
                                << "if (!hEvents[eventCnt]) "
                                   "CUDA_SAFE_CALL(cudaEventCreate(&hEvents[eventCnt]));\n";
                            lu_thread_func_call
                                << "CUDA_SAFE_CALL(cudaEventRecord(hEvents[eventCnt++]));\n";
                        }
                    }
                    lu_thread_func_call << "default_barrier.Notify();\n";

                    if (enable_timing)
                    {
                        lu_thread_func_call << "// Output Timing Result:\n";
                        lu_thread_func_call << "CUDA_SAFE_CALL(cudaDeviceSynchronize());\n";
                        lu_thread_func_call << "float total_ms = 0;\n";
                        lu_thread_func_call
                            << "for (size_t i = 1; i < eventCnt; ++i) { float ms; "
                               "CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, hEvents[i - 1], "
                               "hEvents[i])); "
                               "printf(\"%zd: %.2f ms\\n\", i, ms), total_ms += ms; }\n";
                        lu_thread_func_call
                            << "printf(\" " << thread_name
                            << ": Timing total (except outer memcpy) = %g ms.\\n\", total_ms);\n";
                    }
                    lu_thread_func_call.block_end();
                    // add function call to kernel entry
                    std::string std_thread_func_name =
                        std::string("thread_func") + std::to_string(thread_func_call_count);
                    std::string thread_call_str = std::string("(") + thread_name +
                                                  std::string(", ") + kernel_entry_args +
                                                  std::string(");\n");
                    std::string std_thread_func_call = std::string("auto ") + std_thread_func_name +
                                                       std::string(" = std::bind") +
                                                       thread_call_str;
                    lu_kernel_entry << std_thread_func_call;
                    std::string t_threadpool_call = std::string("schedule_thread_pool->Schedule(");
                    t_threadpool_call += (std_thread_func_name + std::string(");\n"));
                    lu_kernel_entry << t_threadpool_call;
                    ++thread_func_call_count;
                }
                ++thread_index;
            }

            if (thread_kernels_entry.find("default_thread") != thread_kernels_entry.end())
            {
                size_t kernel_order = 0;
                auto& ins_vec = thread_kernels_entry["default_thread"];
                NNFUSION_CHECK(ins_vec.size() > 0);
                int device_id = (*ins_vec[0])["DeviceID"].as<int>();
                lu_kernel_entry << "CUDA_SAFE_CALL(cudaSetDevice(" << device_id << "));\n";
                if (enable_timing)
                {
                    lu_kernel_entry << "static cudaEvent_t hEvents[" << ins_vec.size() + 1
                                    << "];\n";
                    lu_kernel_entry << "size_t eventCnt = 0;\n";
                    lu_kernel_entry << "if (!hEvents[eventCnt]) "
                                       "CUDA_SAFE_CALL(cudaEventCreate(&hEvents[eventCnt]));\n";
                    lu_kernel_entry << "CUDA_SAFE_CALL(cudaEventRecord(hEvents[eventCnt++]));\n";
                }
                for (auto ins : ins_vec)
                {
                    auto kernel = ins->getKernel();
                    auto gnode = ins->getGNode();
                    if (!kernel)
                    {
                        if (ins->name() == "Memcpy")
                        {
                            lu_kernel_entry << " // order=" << ++kernel_order << ", name=memcpy\n";
                            lu_kernel_entry << func_call_codegen(ins)->get_code();
                            if (enable_timing)
                            {
                                lu_kernel_entry
                                    << "if (!hEvents[eventCnt]) "
                                       "CUDA_SAFE_CALL(cudaEventCreate(&hEvents[eventCnt]));\n";
                                lu_kernel_entry
                                    << "CUDA_SAFE_CALL(cudaEventRecord(hEvents[eventCnt++]));\n";
                            }
                        }
                        continue;
                    }
                    const string node_name = gnode ? gnode->get_name() : "internal_node";
                    lu_kernel_entry << " // order=" << ++kernel_order << ", name=" << node_name
                                    << "\n";
                    auto func_call = kernel_func_call[kernel];
                    lu_kernel_entry << func_call_codegen(ins, false, func_call)->get_code();

                    if (enable_debug && !gnode->get_op_ptr()->is_output())
                    {
                        for (size_t i = 0; i < kernel->m_context->outputs.size(); i++)
                        {
                            if (kernel->m_context->outputs[i]->get_element_type().c_type_string() !=
                                "float")
                                continue;
                            auto out_name = kernel->m_context->output_names[i];
                            lu_kernel_entry << "Debug(\"" << node_name << ", " << out_name << "\", "
                                            << out_name << ", \""
                                            << join(kernel->m_context->input_names) << "\", "
                                            << kernel->m_context->outputs[i]->size(false) << ");\n";
                        }
                    }
                    if (enable_timing)
                    {
                        lu_kernel_entry << "if (!hEvents[eventCnt]) "
                                           "CUDA_SAFE_CALL(cudaEventCreate(&hEvents[eventCnt]));\n";
                        lu_kernel_entry
                            << "CUDA_SAFE_CALL(cudaEventRecord(hEvents[eventCnt++]));\n";
                    }
                }

                if (enable_timing)
                {
                    lu_kernel_entry << "// Output Timing Result:\n";
                    lu_kernel_entry << "CUDA_SAFE_CALL(cudaDeviceSynchronize());\n";
                    lu_kernel_entry << "float total_ms = 0;\n";
                    lu_kernel_entry
                        << "for (size_t i = 1; i < eventCnt; ++i) { float ms; "
                           "CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, hEvents[i - 1], hEvents[i])); "
                           "printf(\"%zd: %.2f ms\\n\", i, ms), total_ms += ms; }\n";
                    lu_kernel_entry << "printf(\" default thread: Timing total (except outer "
                                       "memcpy) = %g ms.\\n\", total_ms);\n";
                }
            }

            // destroy cuda stream and event
            if (CUDA_async_manager->num_stream() > 0)
                lu_main_free << CUDA_async_manager->emit_stream_destroy()->get_code();
            if (CUDA_async_manager->num_event() > 0)
                lu_main_free << CUDA_async_manager->emit_event_destroy()->get_code();

            if (CPU_async_manager->num_non_default_stream() > 0)
            {
                lu_main_free << "delete schedule_thread_pool;\n";
            }
            // // destroy handle
            // for (auto& info : dev_cudnn_handle)
            // {
            //     int device_id = info.first;
            //     lu_main_free << "CUDA_SAFE_CALL(cudaSetDevice(" << device_id << "));\n";
            //     for (auto& cudnn_handle : info.second)
            //     {
            //         lu_main_free << "CUDNN_SAFE_CALL(cudnnDestroy(" << cudnn_handle << "));\n";
            //     }
            // }
            // for (auto& info : dev_cublas_handle)
            // {
            //     int device_id = info.first;
            //     lu_main_free << "CUDA_SAFE_CALL(cudaSetDevice(" << device_id << "));\n";
            //     for (auto& cublas_handle : info.second)
            //     {
            //         lu_main_free << "CUBLAS_SAFE_CALL(cublasDestroy(" << cublas_handle << "));\n";
            //     }
            // }
            if (global_required.count("header::super_scaler"))
            {
                lu_kernel_entry << "super_scaler_sync();\n";
                lu_main_free << "super_scaler_finalization();\n";
            }
        }
        for (const auto& allocator : allocator_list)
        {
            lu_main_free << allocator.second->emit_memory_free()->get_code();
        }
        if (CPU_async_manager->num_non_default_stream() > 0)
            lu_kernel_entry << "default_barrier.Wait();\n";
        lu_kernel_entry << "return 0;\n";
        lu_kernel_entry.block_end();
    }

    lu << "\n";
    {
        lu << lu_mem_plan_init.get_code();
        lu << "\nextern \"C\" void cuda_init()";
        lu.block_begin();
        {
            lu << "CUDA_SAFE_CALL(cudaDeviceReset());\n";
            if (global_required.count("header::super_scaler"))
                lu << "super_scaler_initialization();\n";
            // else
            //     lu << "CUDA_SAFE_CALL(cudaSetDevice(0));\n";
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

    if (enable_debug)
    {
        lu << R"(
     inline void Debug(std::string name, float* tensor_ptr, std::string inputs, size_t debug_size = 10, size_t offset=0)
     {
         float* host_tensor = (float*)malloc(sizeof(float) * debug_size);
         CUDA_SAFE_CALL(cudaDeviceSynchronize());
         CUDA_SAFE_CALL(cudaMemcpy(host_tensor, tensor_ptr + offset,  sizeof(float) * debug_size, cudaMemcpyDeviceToHost));
         CUDA_SAFE_CALL(cudaDeviceSynchronize());
         size_t print_size = min((size_t)10, debug_size);
         printf("%s: ", name.c_str());
         for (int i = 0; i < print_size; ++i) printf("%e ", host_tensor[i]);
         printf("...(size= %lu end with %e ) :", debug_size, host_tensor[debug_size - 1]);
         //print with an offset
         size_t print_offset = debug_size / 3;
         print_size = min((size_t)10, debug_size - print_offset);
         for (int i = 0; i < print_size; ++i) printf("%e ", host_tensor[i + print_offset]);
         printf("...(offset= %lu) ", print_offset);
         printf(": %s\n", inputs.c_str());
     }
             )";
    }

    lu << lu_thread_func_call.get_code() << "\n";
    lu << lu_kernel_entry.get_code() << "\n\n";

    // // Test function
    // {
    //     lu << "extern \"C\" int naive_test(";
    //     // Add param
    //     {
    //         vector<string> params;
    //         for (int i = 0; i < tu->arg.size(); i++)
    //         {
    //             auto tv = tu->arg[i];
    //             string type = tv->get_element_type().c_type_string();
    //             stringstream ss;
    //             ss << type << "* " << tv->get_name() << "_host";
    //             params.push_back(ss.str());
    //         }

    //         for (int i = 0; i < tu->out.size(); i++)
    //         {
    //             auto tv = tu->out[i];
    //             string type = tv->get_element_type().c_type_string();
    //             stringstream ss;
    //             ss << type << "* " << tv->get_name() << "_host";
    //             params.push_back(ss.str());
    //         }

    //         lu << join(params, ", ");
    //     }
    //     lu << ")\n";
    //     lu.block_begin();
    //     {
    //         for (size_t i = 0; i < tu->arg.size(); i++)
    //         {
    //             auto& tensor = *tu->arg[i];
    //             lu << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
    //                << ";\n"
    //                << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ","
    //                << tensor.get_tensor_layout()->get_size() << " * "
    //                << tensor.get_element_type().size() << "));\n";

    //             lu << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << ", " << tensor.get_name()
    //                << "_host, " << tensor.get_tensor_layout()->get_size() << " * "
    //                << tensor.get_element_type().size() << ", "
    //                << "cudaMemcpyHostToDevice));\n";
    //         }

    //         lu << "//output arguments\n";
    //         for (size_t i = 0; i < tu->out.size(); i++)
    //         {
    //             auto& tensor = *tu->out[i];
    //             lu << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
    //                << ";\n"
    //                << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ","
    //                << tensor.get_tensor_layout()->get_size() << " * "
    //                << tensor.get_element_type().size() << "));\n";
    //         }

    //         vector<string> params;
    //         for (int i = 0; i < tu->arg.size(); i++)
    //         {
    //             auto& tv = tu->arg[i];
    //             params.push_back(tv->get_name());
    //         }

    //         for (int i = 0; i < tu->out.size(); i++)
    //         {
    //             auto& tv = tu->out[i];
    //             params.push_back(tv->get_name());
    //         }

    //         lu << "naive_entry(" << join(params, ", ") << ");\n";

    //         for (size_t i = 0; i < tu->out.size(); i++)
    //         {
    //             auto& tensor = *tu->out[i];
    //             lu << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << "_host, "
    //                << tensor.get_name() << ", " << tensor.get_tensor_layout()->get_size() << " * "
    //                << tensor.get_element_type().size() << ", "
    //                << "cudaMemcpyDeviceToHost));\n";
    //         }

    //         for (size_t i = 0; i < tu->arg.size(); i++)
    //         {
    //             auto& tensor = *tu->arg[i];
    //             lu << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
    //         }

    //         for (size_t i = 0; i < tu->out.size(); i++)
    //         {
    //             auto& tensor = *tu->out[i];
    //             lu << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
    //         }
    //     }
    //     lu << "return 0;\n";
    //     lu.block_end();
    // }

    // lu << "\n";

    // // Test function 2
    // {
    //     lu << "extern \"C\" int naive_test_simple(void** args)\n";
    //     // Add param
    //     lu.block_begin();
    //     {
    //         lu << "return naive_test(";
    //         vector<string> params;
    //         int acc = 0;
    //         for (int i = 0; i < tu->arg.size(); i++, acc++)
    //         {
    //             auto tv = tu->arg[i];
    //             string type = tv->get_element_type().c_type_string();
    //             stringstream ss;
    //             ss << "(" << type << "*)args[" << acc << "]";
    //             params.push_back(ss.str());
    //         }

    //         for (int i = 0; i < tu->out.size(); i++, acc++)
    //         {
    //             auto tv = tu->out[i];
    //             string type = tv->get_element_type().c_type_string();
    //             stringstream ss;
    //             ss << "(" << type << "*)args[" << acc << "]";
    //             params.push_back(ss.str());
    //         }
    //         lu << join(params, ", ");
    //         lu << ");\n";
    //     }
    //     lu.block_end();
    // }

    // generate main() function
    std::string function_include =
        "#include \"nnfusion_rt.h\"\n#include <stdlib.h>\n#include <stdio.h>\n";
    LanguageUnit h2dcopy("h2dcopy");
    LanguageUnit d2hcopy("d2hcopy");
    LanguageUnit fillval("fillval");

    LanguageUnit& lu_main = *this->lu_main;
    {
        lu_main << function_include << "\n";
        lu_main << header::stdexcept->get_code();
        lu_main << "#include <sstream>\n";
        lu_main << "#include <cuda_profiler_api.h>\n";
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
                        << "_host, *" << tensor.get_name() << ";\n";

                lu_main << "CUDA_SAFE_CALL(cudaMallocHost((void**)&" << tensor.get_name()
                        << "_host, sizeof(" << tensor.get_element_type().c_type_string() << ")* "
                        << tensor.get_tensor_layout()->get_size() << "));\n";
                lu_main << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ", "
                        << "sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                        << tensor.get_tensor_layout()->get_size() << "));\n";

                fillval << "for (int i = 0; i < " << tensor.get_tensor_layout()->get_size()
                        << "; ++i) " << tensor.get_name() << "_host[i] = 1.0f;\n";

                h2dcopy << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << ", "
                        << tensor.get_name() << "_host, "
                        << "sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                        << tensor.get_tensor_layout()->get_size() << ", "
                        << "cudaMemcpyHostToDevice));\n";
            }

            lu_main << "\n//output arguments\n";
            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                //malloc host output arg
                lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                        << "_host, *" << tensor.get_name() << ";\n";

                lu_main << "CUDA_SAFE_CALL(cudaMallocHost((void**)&" << tensor.get_name()
                        << "_host, sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                        << tensor.get_tensor_layout()->get_size() << "));\n";

                d2hcopy << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << "_host, "
                        << tensor.get_name() << ", "
                        << " sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                        << tensor.get_tensor_layout()->get_size() << ", "
                        << "cudaMemcpyDeviceToHost));\n";
            }

            lu_main << "\n// fill input values\n";
            lu_main << fillval.get_code() << "\n";
            lu_main << h2dcopy.get_code() << "\n";

            vector<string> params;
            for (int i = 0; i < tu->arg.size(); i++)
            {
                auto& tv = tu->arg[i];
                params.push_back(tv->get_name());
            }
            for (int i = 0; i < tu->out.size(); i++)
            {
                auto& tv = tu->out[i];
                params.push_back("&" + tv->get_name());
            }
            int warm_step = 5, test_step = 100;
            if (enable_debug)
            {
                warm_step = 0;
                test_step = 1;
            }
            lu_main << "\n//warm up for 5 iters:\n";
            lu_main << "for(int i_=0; i_< " << warm_step << "; i_++)\n";
            lu_main.block_begin();
            lu_main << h2dcopy.get_code();
            lu_main << "kernel_entry(" << join(params, ", ") << ");\n";
            lu_main << d2hcopy.get_code();
            lu_main << "CUDA_SAFE_CALL(cudaDeviceSynchronize()); \n";

            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                lu_main << "printf(\"%s \\n\", \"" << tensor.get_name() << ":\");\n"
                        << "for (int i = 0; i < "
                        << std::min(size_t(10), tensor.get_tensor_layout()->get_size())
                        << "; ++i) printf(\"%e \", (float)" << tensor.get_name() << "_host[i]); "
                        << "\nprintf(\" .. (size = " << tensor.get_tensor_layout()->get_size()
                        << ", ends with %e);\\n\", (float)" << tensor.get_name() << "_host["
                        << tensor.get_tensor_layout()->get_size() - 1 << "]);\n";
            }
            lu_main.block_end();

            lu_main << "\n//GPU time measurement\n";
            lu_main << "cudaEvent_t start, stop;\n";
            lu_main << "cudaEventCreate(&start);\n";
            lu_main << "cudaEventCreate(&stop);\n";

            lu_main << "\n//time measurement\n";
            lu_main << "cudaEventRecord(start);\n\n";
            lu_main << "//kernel call\n";

            lu_main << "int steps = " << test_step << ";\n";
            lu_main << "cudaProfilerStart();\n";
            lu_main << "for (int i_=0; i_<steps; i_++)\n";
            lu_main.block_begin();

            lu_main << h2dcopy.get_code();
            // kernel launch
            lu_main << "kernel_entry(" << join(params, ", ") << ");\n";

            // lu_main << d2hcopy.get_code();

            lu_main.block_end();
            lu_main << "cudaProfilerStop();\n";

            lu_main << "//time measurement\n";
            lu_main << "\ncudaEventRecord(stop);\n";
            lu_main << "cudaEventSynchronize(stop);\n";
            lu_main << "float milliseconds = 0;\n";
            lu_main << "cudaEventElapsedTime(&milliseconds, start, stop);\n";
            lu_main << "printf(\"function execution time: %f ms\\n\", milliseconds/steps);\n";
            lu_main << "\n//free context\n";

            for (size_t i = 0; i < tu->arg.size(); i++)
            {
                auto& tensor = *tu->arg[i];
                lu_main << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
            }

            lu_main << "cuda_free();\n\n";
        }

        //free host input args
        for (size_t i = 0; i < tu->arg.size(); i++)
        {
            auto& tensor = *tu->arg[i];
            lu_main << "cudaFreeHost(" << tensor.get_name() << "_host);\n";
        }
        //free host output args
        for (size_t i = 0; i < tu->out.size(); i++)
        {
            auto& tensor = *tu->out[i];
            lu_main << "cudaFreeHost(" << tensor.get_name() << "_host);\n";
        }

        lu_main << "return 0;\n";
        lu_main.block_end();
    }

    //generate include header file
    lu_include << "// Microsoft (c) 2019\n";
    lu_include << "#pragma once\n";
    lu_include << declaration::typedef_int->get_code() << "\n";
    lu_include << lu_kernel_entry_header.get_code() << ";\n";
    lu_include << "extern \"C\" void cuda_init();\n";
    lu_include << "extern \"C\" void cuda_free();\n";
    lu_include << header::cuda->get_code();

    //generate CMakeList.txt
    LanguageUnit& lu_cmake = *this->lu_cmakefile;
    super_scaler_enable = global_required.count("header::super_scaler") > 0;
    lu_cmake << get_generate_cmakelists();
    if (CPU_async_manager->num_non_default_stream() > 0)
    {
        lu_cmake << "# need to specify the correct path of eigen\n"
                 << "set(EIGEN_DIR \"/usr/include/eigen3\" CACHE STRING \"EIGEN libraries folder "
                    "location\")\n"
                 << "include_directories(${EIGEN_DIR})\n\n";

        char exe_path[PATH_MAX];
        size_t count = readlink("/proc/self/exe", exe_path, PATH_MAX);
        const char* path;
        if (count != -1)
        {
            path = dirname(exe_path);
        }
        else
        {
            throw std::runtime_error("Failed to get the directory of executable file.\n");
        }

        std::string threadpool_path = std::string(path) + std::string("/threadpool");
        std::string cmd = std::string("cp -R ") + threadpool_path + std::string(" .");
        if (0 != system(cmd.c_str()))
        {
            throw std::runtime_error("Failed to copy threadpool source files.\n");
        }
        lu_cmake << "include(threadpool/threadpool.cmake)\n";
        lu_cmake << "target_link_libraries(nnfusion_naive_rt threadpool)\n\n";
    }

    post_projgen();
    projgen();
    after_projgen();

    // change to working directory
    int status = chdir("../../");
    if (ctx->m_graphs.size() > 1)
        status = chdir("../");
    return rc;
}
