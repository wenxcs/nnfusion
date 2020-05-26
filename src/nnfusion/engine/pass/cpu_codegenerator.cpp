// Microsoft (c) 2019, NNFusion Team

#include <libgen.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "codegenerator_helper.hpp"
#include "cpu_codegenerator.hpp"
#include "nnfusion/core/kernels/cpu/cpu_kernel_emitter.hpp"
#include "nnfusion/core/kernels/cpu/cpu_langunit.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/engine/async_manager.hpp"
#include "nnfusion/engine/memory_allocator.hpp"
// For reference kernels
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/core/kernels/cpu/barrier.hpp"
#include "nnfusion/core/kernels/cpu/reference/reference_common.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::async;

DEFINE_int32(fnuma_node_num, 1, "");
DEFINE_int32(fthread_num_per_node, 0, "");
DECLARE_bool(fkernels_as_files);
DECLARE_int64(fkernels_files_number);
DECLARE_bool(frt_const_folding);

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
                NNFUSION_LOG(INFO) << "Error creating directory: " + tar_path;
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

bool CpuCodeGenerator::setpwd(std::shared_ptr<InterpreterContext> ctx,
                              std::shared_ptr<TranslationUnit> tu)
{
    std::string working_dir = "./nnfusion_rt";
    create_dir(working_dir);
    std::string tar_path = working_dir + "/cpu_codegen/";
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

bool CpuCodeGenerator::projgen()
{
    save_file(this->lu_cmakefile);
    save_file(this->lu_nnfusion_rt);
    save_file(this->lu_header);
    save_file(this->lu_main);
    return true;
}

std::pair<std::string, std::string> CpuCodeGenerator::get_paras_and_args(
    std::vector<nnfusion::kernels::KernelEmitter::Pointer>& kernel_vec)
{
    std::pair<std::string, std::string> paras_and_args;
    vector<string> params;
    vector<string> args;
    unordered_set<string> allocated;
    for (auto kernel : kernel_vec)
    {
        if (kernel && kernel->m_context)
        {
            for (auto input : kernel->m_context->inputs)
            {
                auto name = input->get_name();
                if (allocated.find(name) == allocated.end() &&
                    name.compare(0, 10, "Parameter_") == 0)
                {
                    string type = input->get_element_type().c_type_string();
                    stringstream ss;
                    ss << type << "* " << name;
                    allocated.insert(name);
                    params.push_back(ss.str());
                    args.push_back(name);
                }
            }
            if (kernel->m_context->gnode && kernel->m_context->gnode->get_op_ptr()->is_output())
            {
                for (auto output : kernel->m_context->outputs)
                {
                    auto name = output->get_name();
                    if (allocated.find(name) == allocated.end())
                    {
                        string type = output->get_element_type().c_type_string();
                        stringstream ss;
                        ss << type << "* " << name;
                        allocated.insert(name);
                        params.push_back(ss.str());
                        args.push_back(name);
                    }
                }
            }
        }
    }
    paras_and_args.first = join(params, ", ");
    paras_and_args.second = join(args, ", ");
    return paras_and_args;
}

bool CpuCodeGenerator::run(std::shared_ptr<InterpreterContext> ctx,
                           std::shared_ptr<TranslationUnit> tu)
{
    setpwd(ctx, tu);

    NNFUSION_CHECK_NOT_NULLPTR(tu->memory_allocator_factory);
    auto& allocator_list = tu->memory_allocator_factory->get_allocator_list();
    auto async_manager = AsyncManagerFactory::get_host_async_manager(tu->m_graph, GENERIC_CPU);

    this->lu_cmakefile = LanguageUnit_p(new LanguageUnit("CMakeLists.txt"));
    this->lu_nnfusion_rt = LanguageUnit_p(new LanguageUnit("nnfusion_rt.cpp"));
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
    bool need_intra_node_threadpool = false;

    //std::vector<shared_ptr<KernelEmitter>> kernels;
    auto& prog = tu->program;
    for (auto iterator : prog)
    {
        for (auto ins : *iterator)
        {
            if (ins->name() == "Memcpy" || (ins->getGNode() && ins->getGNode()->is_parameter()))
            {
                continue;
            }
            auto kernel = ins->getKernel();
            if (kernel && kernel->get_or_emit_source())
            {
                need_intra_node_threadpool |= kernel->is_parallelism();
            }
            else
            {
                shared_ptr<const KernelRegistration> kernel_reg =
                    KernelRegistry::Global()->FindKernelRegistration(
                        "AnyOP", GENERIC_CPU, DT_FLOAT);
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
        re.require(header::fstream);
        re.require(header::thread);
        // Both intra_node parallelism and multi-stream need worker_thread_pool.
        if (need_intra_node_threadpool || async_manager->num_non_default_stream() > 0)
        {
            re.require(header::threadpool);
            re.require(declaration::worker_thread_pool);
        }
        if (async_manager->num_non_default_stream() > 0)
            re.require(declaration::schedule_thread_pool);
        if (async_manager->num_event() > 0 || async_manager->num_non_default_stream() > 0)
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

    lu << "#include \"nnfusion_rt.h\"\n";
    unordered_map<string, LanguageUnit_p> decleard_function_LU;
    // Collect Function Definition
    {
        vector<codegenerator::CPUFunctionFile> cpu_kernel_files;
        if (FLAGS_fkernels_as_files && FLAGS_fkernels_files_number > 0)
            cpu_kernel_files.resize(FLAGS_fkernels_files_number);
        int cpu_kernel_n = 0;
        LanguageUnit def("FUNCTIONS");
        for (auto iterator : prog)
        {
            for (auto ins : *iterator)
            {
                auto kernel = ins->getKernel();
                auto gnode = ins->getGNode();
                if (!kernel)
                    continue;
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
                string func_key = fu->signature_unit->get_code() + body_unit;
                if (kernel->is_static_function() ||
                    decleard_function_LU.find(func_key) == decleard_function_LU.end())
                {
                    auto functionfile = codegenerator::CPUFunctionFile::convert_from(kernel);
                    if (FLAGS_fkernels_as_files)
                    {
                        def << functionfile->get_extern_declare();
                        if (FLAGS_fkernels_files_number > 0)
                            cpu_kernel_files[cpu_kernel_n].merge_from(functionfile);
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
                    // def << "// Function declared:" << fu->body_unit->symbol << "\n\n";
                }
                if (FLAGS_fkernels_files_number > 0)
                {
                    cpu_kernel_n++;
                    cpu_kernel_n %= FLAGS_fkernels_files_number;
                }
            }
        }
        if (FLAGS_fkernels_as_files && FLAGS_fkernels_files_number > 0)
        {
            for (int i = 0; i < FLAGS_fkernels_files_number; i++)
                cpu_kernel_files[i].save_file();
        }

        //Write Dependency
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("header::") != string::npos)
                lu << it.second->get_code();
        lu << "#include <cstring>\n";
        lu << "using namespace std;\n";
        lu << "\n";
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("macro::") != string::npos)
                lu << it.second->get_code();
        lu << "\n";
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("declaration::") != string::npos)
                lu << it.second->get_code();
        lu << "\n";
        // stream and event declaration
        if (async_manager->num_stream() > 0)
            lu << async_manager->emit_stream_decl()->get_code();
        if (async_manager->num_event() > 0)
            lu << async_manager->emit_event_decl()->get_code();
        // default barrier declaration
        if (async_manager->num_non_default_stream() > 0)
        {
            lu << "nnfusion::cpu::Notification init_barrier;\n";
            lu << "nnfusion::cpu::Barrier default_barrier("
               << async_manager->num_non_default_stream() << ");\n";
        }
        /*
        {
            //hard coded order
            if(re.local_symbol.find("cpu_reference_sum") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_product") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_max") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_min") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_softmax") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_broadcast") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_argmax") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_argmin") != re.local_symbol.end()
            )
            {
                re.local_symbol.erase("cpu_reference_reduce");
                //
            }
        }
        */
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("cpu_reference_") != string::npos)
                lu << it.second->get_code();
        lu << "\n";

        //Write Code
        lu << def.collect_code() << "\n";

        if (re.local_symbol.find("header::reference_common") != re.local_symbol.end())
        {
            save_file(reference_common_header);
        }
        if (async_manager->num_event() > 0 || async_manager->num_non_default_stream() > 0)
            save_file(barrier_header);
    }

    // Generate caller function body
    {
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
                ss << type << "* " << tv->get_name();
                allocated.insert(tv->get_name());
                params.push_back(ss.str());
                args.push_back(tv->get_name());
            }
            kernel_entry_params = join(params, ", ");
            kernel_entry_args = join(args, ", ");

            lu_kernel_entry << kernel_entry_params;
        }

        lu_kernel_entry << ")";
        lu_kernel_entry_header << lu_kernel_entry.get_code();
        lu_kernel_entry << "\n";
        lu_kernel_entry.block_begin();

        // reset event/notification
        lu_kernel_entry << async_manager->emit_event_reset()->get_code();
        if (async_manager->num_non_default_stream() > 0)
            lu_kernel_entry << "default_barrier.Reset();\n";

        //Planning
        for (const auto& allocator : allocator_list)
        {
            lu_mem_plan_init << allocator.second->emit_memory_init()->get_code();
            lu_main_init << allocator.second->emit_memory_alloc()->get_code();
        }

        //Function Call
        {
            /*
            if (global_required.count("declaration::eigen_global_thread_pool") > 0)
            {
                if (FLAGS_fthread_num_per_node == 0)
                {
                    lu_main_init
                        << "int thread_count = std::thread::hardware_concurrency() >> 1;\n";
                }
                else
                {
                    lu_main_init << "int thread_count = " << FLAGS_fthread_num_per_node << ";\n";
                }
                lu_main_init << "global_thread_pool = new Eigen::ThreadPool(thread_count? "
                                "thread_count : 1);\n";
            }
            if (global_required.count("declaration::eigen_global_thread_pool_device") > 0)
            {
                lu_main_init << "global_thread_pool_device = new "
                                "Eigen::ThreadPoolDevice(global_thread_pool, "
                                "global_thread_pool->NumThreads());\n";
            }
	    */

            if (global_required.count("header::eigen_spatial_convolution") > 0)
            {
                lu_main_init << "setenv(\"OMP_NUM_THREADS\", \"1\", true);\n";
            }

            // Both intra_node parallelism and multi-thread need worker_thread_pool.
            // If multi-thread is not enabled, numa-aware can be ignored.
            int numa_node_num = FLAGS_fnuma_node_num;
            if (async_manager->num_non_default_stream() > 0)
            {
                lu_main_init << "schedule_thread_pool = new concurrency::NumaAwareThreadPool();\n";
            }
            else
            {
                numa_node_num = 1;
            }

            if (need_intra_node_threadpool || async_manager->num_non_default_stream() > 0)
            {
                lu_main_init << "worker_thread_pool = new concurrency::NumaAwareThreadPool("
                             << numa_node_num << ", " << FLAGS_fthread_num_per_node << ");\n";
            }

            std::unordered_map<string, vector<shared_ptr<KernelEmitter>>> stream_kernels_entry;
            std::unordered_map<shared_ptr<KernelEmitter>, string> kernel_func_name;
            for (auto iterator : prog)
            {
                for (auto ins : *iterator)
                {
                    auto kernel = ins->getKernel();
                    auto gnode = ins->getGNode();
                    auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
                    auto stream = async_info.execution_thread;

                    if (!kernel)
                        continue;
                    FunctionUnit_p fu = kernel->get_or_emit_source(true);
                    std::string func_name;
                    if (kernel->is_static_function())
                    {
                        func_name = fu->name_unit->get_code();
                    }
                    else
                    {
                        std::string body_unit = fu->body_unit->get_code();
                        string func_key = fu->signature_unit->get_code() + body_unit;
                        NNFUSION_CHECK(decleard_function_LU.find(func_key) !=
                                       decleard_function_LU.end());
                        func_name = decleard_function_LU[func_key]->get_code();
                    }

                    kernel_func_name[kernel] = func_name;
                    if (gnode->is_constant() || gnode->is_variable() ||
                        (FLAGS_frt_const_folding && (*ins)["rt_const_folding"].is_valid_as<bool>()))
                    {
                        if (!kernel->is_eliminative())
                        {
                            NNFUSION_CHECK(stream->is_default_stream())
                                << "Kernel function calls in cpu_init() "
                                   "should use default/main stream/thread.";
                            lu_main_init << kernel_func_name[kernel];
                            string call_str = fu->call_unit->get_code();
                            if (kernel->is_parallelism())
                            {
                                std::string threadpool_param =
                                    "worker_thread_pool->GetRawThreadPool(), ";
                                call_str.insert(1, threadpool_param);
                                lu_main_init << call_str;
                            }
                            else
                            {
                                lu_main_init << fu->call_unit->get_code();
                            }
                        }
                        else
                        {
                            lu_main_init << " // eliminated\n";
                        }
                    }
                    // organize kernels according to their streams/threads
                    else
                    {
                        std::string stream_name =
                            stream->is_default_stream() ? "default" : stream->get_name();
                        stream_kernels_entry[stream_name].push_back(kernel);
                    }
                }
            }
            if (async_manager->num_non_default_stream() > 0)
            {
                lu_main_init << "init_barrier.Notify();\n";
                lu_kernel_entry << "init_barrier.Wait();\n";
            }
            // emit function calls in kernel_entry()
            int stream_index = 0;
            int thread_func_call_count = 1;
            for (auto& sk : stream_kernels_entry)
            {
                auto& stream_name = sk.first;
                if (stream_name != "default")
                {
                    auto thread_call_paras = get_paras_and_args(sk.second).first;
                    auto thread_call_args = get_paras_and_args(sk.second).second;
                    if (!thread_call_args.empty())
                        thread_call_args = ", " + thread_call_args;
                    // add thread_calls definition
                    lu_thread_func_call << "extern \"C\" void " << stream_name << "_Call(";
                    lu_thread_func_call << thread_call_paras << ")\n";
                    lu_thread_func_call.block_begin();
                    int func_call_count = 1;
                    for (auto kernel : sk.second)
                    {
                        auto gnode = kernel->m_context->gnode;
                        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
                        if (!async_info.wait_barriers.empty())
                        {
                            for (auto event : async_info.wait_barriers)
                            {
                                lu_thread_func_call
                                    << async_manager
                                           ->emit_event_wait(async_info.execution_thread, event)
                                           ->get_code();
                            }
                        }
                        if (!kernel->is_eliminative())
                        {
                            // For mlas kernels, the function call is:
                            //   kernel_func_name(worker_thread_pool->GetRawThreadPool(numa_node_id), param1, param2, ...);
                            // For other kernels, the function call is:
                            //   auto func1 = std::bind(kernel_func_name, param1, param2, ...);
                            //   worker_thread_pool->ScheduleSync(func1, numa_node_id);
                            int numa_node = stream_index % numa_node_num;
                            FunctionUnit_p fu = kernel->get_or_emit_source(true);
                            string call_str = fu->call_unit->get_code();
                            if (kernel->is_parallelism())
                            {
                                std::string threadpool_param =
                                    "worker_thread_pool->GetRawThreadPool(";
                                threadpool_param +=
                                    (std::to_string(numa_node) + std::string("), "));
                                call_str.insert(1, threadpool_param);
                                lu_thread_func_call << kernel_func_name[kernel];
                                lu_thread_func_call << call_str;
                            }
                            else
                            {
                                call_str.insert(1, kernel_func_name[kernel] + std::string(", "));
                                std::string std_func_name =
                                    std::string("func") + std::to_string(func_call_count);
                                std::string std_func_call = std::string("auto ") + std_func_name +
                                                            std::string(" = std::bind") + call_str;
                                lu_thread_func_call << std_func_call;
                                std::string threadpool_call =
                                    std::string("worker_thread_pool->ScheduleSync(");
                                threadpool_call += (std_func_name + std::string(", ") +
                                                    std::to_string(numa_node) + ");\n");
                                lu_thread_func_call << threadpool_call;
                                ++func_call_count;
                            }
                        }
                        else
                        {
                            lu_thread_func_call << " // eliminated\n";
                        }
                        if (async_info.notify_barrier != nullptr)
                        {
                            lu_thread_func_call
                                << async_manager->emit_event_record(async_info.notify_barrier)
                                       ->get_code();
                        }
                    }
                    lu_thread_func_call << "default_barrier.Notify();\n";
                    lu_thread_func_call.block_end();
                    // add function call to kernel entry
                    std::string std_thread_func_name =
                        std::string("thread_func") + std::to_string(thread_func_call_count);
                    std::string thread_call_str = std::string("(") + stream_name +
                                                  std::string("_Call") + thread_call_args +
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
                ++stream_index;
            }

            if (stream_kernels_entry.find("default") != stream_kernels_entry.end())
            {
                for (auto kernel : stream_kernels_entry["default"])
                {
                    FunctionUnit_p fu = kernel->get_or_emit_source(true);

                    {
                        auto gnode = kernel->m_context->gnode;
                        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
                        if (!async_info.wait_barriers.empty())
                        {
                            for (auto event : async_info.wait_barriers)
                            {
                                lu_kernel_entry
                                    << async_manager
                                           ->emit_event_wait(async_info.execution_thread, event)
                                           ->get_code();
                            }
                        }
                        if (!kernel->is_eliminative())
                        {
                            // lu_kernel_entry << "t_start= Clock::now();\n";
                            // lu_kernel_entry << "for (int i = 0; i < test_time; ++i){\n";

                            lu_kernel_entry << kernel_func_name[kernel];

                            string call_str = fu->call_unit->get_code();
                            if (kernel->is_parallelism())
                            {
                                std::string threadpool_param =
                                    "worker_thread_pool->GetRawThreadPool(), ";
                                call_str.insert(1, threadpool_param);
                                lu_kernel_entry << call_str;
                            }
                            else
                            {
                                lu_kernel_entry << fu->call_unit->get_code();
                            }

                            // lu_kernel_entry << "}\n";
                            // lu_kernel_entry << "t_end= Clock::now();\n";
                            // lu_kernel_entry << "fp_ms= (t_end- t_start)/test_time;\n";
                            // lu_kernel_entry << "printf(\"{\\\"" << gnode->get_name() <<"\\\",%f},\", fp_ms.count());\n";
                        }
                        else
                        {
                            lu_kernel_entry << " // eliminated;\n";
                        }
                        if (async_info.notify_barrier != nullptr)
                        {
                            lu_kernel_entry
                                << async_manager->emit_event_record(async_info.notify_barrier)
                                       ->get_code();
                        }
                    }
                }
            }
        }

        for (const auto& allocator : allocator_list)
        {
            lu_main_free << allocator.second->emit_memory_free()->get_code();
        }
        /*
        if (global_required.count("declaration::eigen_global_thread_pool") > 0)
        {
            lu_main_free << "free(global_thread_pool);\n";
        }
        if (global_required.count("declaration::eigen_global_thread_pool_device") > 0)
        {
            lu_main_free << "free(global_thread_pool_device);\n";
        }
	*/
        if (need_intra_node_threadpool || async_manager->num_non_default_stream() > 0)
        {
            lu_main_free << "delete worker_thread_pool;\n";
        }
        if (async_manager->num_non_default_stream() > 0)
        {
            lu_main_free << "delete schedule_thread_pool;\n";
        }

        if (async_manager->num_non_default_stream() > 0)
            lu_kernel_entry << "default_barrier.Wait();\n";
        lu_kernel_entry << "\nreturn 0;\n";
        lu_kernel_entry.block_end();
    }

    lu << "\n";
    {
        lu << lu_mem_plan_init.get_code();
        lu << "\nextern \"C\" void cpu_init()";
        lu.block_begin();
        {
            lu << lu_main_init.get_code();
        }
        lu.block_end();
        lu << "\n";

        lu << "extern \"C\" void cpu_free()";
        lu.block_begin();
        {
            lu << lu_main_free.get_code();
        }

        lu.block_end();
    }
    lu << "\n";
    lu << lu_thread_func_call.get_code() << "\n";
    lu << lu_kernel_entry.get_code() << "\n\n";

    // generate main() function
    std::string function_include =
        R"(
#include <chrono>
#include <stdio.h>      
#include <stdlib.h>
#include "nnfusion_rt.h"
)";
    LanguageUnit fillval("fillval");

    LanguageUnit& lu_main = *this->lu_main;
    {
        lu_main << function_include << "\n";
        lu_main << header::stdexcept->get_code();
        lu_main << "#include <sstream>\n";
        lu_main << "\n";
        lu_main << "using Clock = std::chrono::high_resolution_clock;\n\n";

        lu_main << "int main(void)";
        lu_main.block_begin();
        {
            lu_main << "\ncpu_init();\n\n";

            for (size_t i = 0; i < tu->arg.size(); i++)
            {
                auto& tensor = *tu->arg[i];
                //malloc host input arg
                lu_main << "//input argument\n";
                lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                        << "_host = (" << tensor.get_element_type().c_type_string() << "*)"
                        << "malloc( sizeof(" << tensor.get_element_type().c_type_string() << ")* "
                        << tensor.get_tensor_layout()->get_size() << ");\n";

                fillval << "for (int i = 0; i < " << tensor.get_tensor_layout()->get_size()
                        << "; ++i) " << tensor.get_name() << "_host[i] = 1.0f;\n";
            }

            lu_main << "\n//output arguments\n";
            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                //malloc host output arg
                lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                        << "_host = (" << tensor.get_element_type().c_type_string() << "*)"
                        << "malloc( sizeof(" << tensor.get_element_type().c_type_string() << ")* "
                        << tensor.get_tensor_layout()->get_size() << ");\n ";
            }
            lu_main << "\n//fill input values\n";
            lu_main << fillval.get_code();

            vector<string> params;
            for (int i = 0; i < tu->arg.size(); i++)
            {
                auto& tv = tu->arg[i];
                params.push_back(tv->get_name() + "_host");
            }
            for (int i = 0; i < tu->out.size(); i++)
            {
                auto& tv = tu->out[i];
                params.push_back(tv->get_name() + "_host");
            }

            lu_main << "\n//warm up\n";
            lu_main << "int warm_steps = 5;\n";
            lu_main << "for(int i_=0; i_<warm_steps; i_++)\n";
            lu_main.block_begin();
            // kernel launch
            lu_main << "kernel_entry(" << join(params, ", ") << ");\n";
            lu_main.block_end();

            lu_main << "\n//time measurement\n";
            lu_main << "auto t_start = Clock::now();\n\n";
            lu_main << "//kernel call\n";
            lu_main << "int test_steps = 100;\n";
            lu_main << "for(int i_=0; i_<test_steps; i_++)\n";
            lu_main.block_begin();
            // kernel launch
            lu_main << "kernel_entry(" << join(params, ", ") << ");\n";
            lu_main.block_end();

            lu_main << "\n//time measurement\n";
            lu_main << "auto t_end = Clock::now();\n";
            lu_main << "std::chrono::duration<double, std::milli> fp_ms = t_end - t_start;\n\n";

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

            lu_main
                << "\nprintf(\"function execution time: %f ms\\n\", fp_ms.count()/test_steps);\n";
            lu_main << "\n//free context\n";
            lu_main << "cpu_free();\n";
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

        lu_main << "\nreturn 0;\n";
        lu_main.block_end();
    }

    //generate include header file
    lu_include << "// Microsoft (c) 2019\n";
    lu_include << "#pragma once\n";
    lu_include << declaration::typedef_int->get_code() << "\n";
    lu_include << lu_kernel_entry_header.get_code() << ";\n";
    lu_include << "extern \"C\" void cpu_init();\n";
    lu_include << "extern \"C\" void cpu_free();\n";

    //generate CMakeList.txt
    LanguageUnit& lu_cmake = *this->lu_cmakefile;
    //lu_cmake << generate_cmakelists();
    lu_cmake << "project(main_test)\n"
             << "cmake_minimum_required(VERSION 3.5)\n\n";

    if (global_required.count("header::cblas") > 0)
    {
        lu_cmake << R"(
set(NNFUSION_THIRDPARTY_FOLDER "~/repo/Thirdparty" CACHE STRING "NNFusion Thirdpary libraries folder location")
if(EXISTS "${NNFUSION_THIRDPARTY_FOLDER}")
else()
message(SEND_ERROR "NNFUSION_THIRDPARTY_FOLDER not exists." )
endif()
# include(mkldnn.cmake)
set(MKL_LIBS libiomp5.so libmklml_intel.so)
set(MKL_ROOT ${NNFUSION_THIRDPARTY_FOLDER}/mkl/mkl_lnx)
add_library(libmkl INTERFACE)
foreach(LIB ${MKL_LIBS})
    target_link_libraries(libmkl INTERFACE ${MKL_ROOT}/lib/${LIB})
endforeach()
        )"
                 << "\n";
    }

    lu_cmake
        << "set (CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -std=gnu++11 -O3 -march=native -pthread\")\n"
        << (FLAGS_fkernels_as_files ? "file(GLOB kernels "
                                      "kernels/*.cpp)\nadd_library(nnfusion_cpu_rt "
                                      "nnfusion_rt.cpp ${kernels})\n"
                                    : "add_library(nnfusion_cpu_rt nnfusion_rt.cpp)\n");
    if (global_required.count("header::cblas") > 0)
    {
        lu_cmake << "target_link_libraries(nnfusion_cpu_rt pthread libmkl)\n\n";
    }

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

    lu_cmake << "find_package(Threads REQUIRED)\n";
    lu_cmake << "target_link_libraries(nnfusion_cpu_rt Threads::Threads)\n\n";

    if (need_intra_node_threadpool || async_manager->num_non_default_stream() > 0)
    {
        // Prepare eigen submodule.
        std::string eigen_path = std::string(path) + std::string("/eigen");
        std::string cmd = std::string("cp -R ") + eigen_path + std::string(" .");
        if (0 != system(cmd.c_str()))
        {
            throw std::runtime_error("Failed to copy eigen source files.\n");
        }
        lu_cmake << "include(eigen/eigen.cmake)\n";
        lu_cmake << "target_link_libraries(nnfusion_cpu_rt eigen)\n\n";

        // Prepare threadpool submodule.
        std::string threadpool_path = std::string(path) + std::string("/threadpool");
        cmd = std::string("cp -R ") + threadpool_path + std::string(" .");
        if (0 != system(cmd.c_str()))
        {
            throw std::runtime_error("Failed to copy threadpool source files.\n");
        }
        lu_cmake << "include(threadpool/threadpool.cmake)\n";
        lu_cmake << "target_link_libraries(nnfusion_cpu_rt threadpool)\n\n";
    }

    if (global_required.count("header::mlas") > 0)
    {
        // Prepare mlas submodule.
        std::string mlas_path = std::string(path) + std::string("/mlas");
        std::string cmd = std::string("cp -R ") + mlas_path + std::string(" .");
        if (0 != system(cmd.c_str()))
        {
            throw std::runtime_error("Failed to copy mlas source files.\n");
        }
        lu_cmake << "include(mlas/mlas.cmake)\n";
        lu_cmake << "target_link_libraries(nnfusion_cpu_rt mlas)\n\n";
    }

    lu_cmake << "target_compile_options(nnfusion_cpu_rt PRIVATE \"-fPIC\")\n"
             << "add_executable(main_test main_test.cpp)\n"
             << "target_link_libraries(main_test nnfusion_cpu_rt)\n";

    lu_cmake << R"(
if(EXISTS "${CMAKE_BINARY_DIR}/Constant")
else()
add_custom_command(
    TARGET nnfusion_cpu_rt
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/Constant ${CMAKE_BINARY_DIR}/Constant
)
endif()
)";

    projgen();

    // change to working directory
    int status = chdir("../../");
    if (ctx->m_graphs.size() > 1)
        status = chdir("../");
    return rc;
}
