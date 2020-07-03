// Microsoft (c) 2019, Wenxiang Hu

#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/engine/async_manager.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
#include "cuda_codegenerator.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/op.hpp"

DECLARE_bool(fkernels_as_files);

namespace nnfusion
{
    class RocmCodeGenerator : public CudaCodeGenerator
    {
    public:
        NNFusion_DeviceType device_type() override { return NNFusion_DeviceType::ROCM_GPU; }
        virtual std::string get_generate_cmakelists(void) override
        {
            LanguageUnit lu;
            lu << R"(project(main_test)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
set(CMAKE_CXX_FLAGS "-O2 -Wno-ignored-attributes -Wno-duplicate-decl-specifier")
)" << (super_scaler_enable ? "find_package(MPI)" : "")
               << R"(
include_directories(
    /opt/rocm/include
    /opt/rocm/rocblas/include
    /opt/rocm/rocrand/include
    /opt/rocm/hiprand/include
    /opt/rocm/hipsparse/include
)" << (super_scaler_enable ? "${MPI_INCLUDE_PATH}" : "")
               << R"(
)
)" << (super_scaler_enable
           ? "find_library(ssrocm libsuper_scaler_rocm.so ${CMAKE_CURRENT_SOURCE_DIR})"
           : "")
               << (FLAGS_fkernels_as_files
                       ? "file(GLOB kernels kernels/*.cpp)\nadd_library(nnfusion_naive_rt "
                         "nnfusion_rt.cpp ${kernels})\n"
                       : "add_library(nnfusion_naive_rt nnfusion_rt.cpp)\n")
               << R"(
add_executable(main_test main_test.cpp)
target_link_libraries(main_test nnfusion_naive_rt /opt/rocm/lib/libMIOpen.so /opt/rocm/lib/librocblas.so )"
               << (super_scaler_enable ? "${ssrocm} ${MPI_LIBRARIES}" : "") << R"()
if(EXISTS "${CMAKE_BINARY_DIR}/Constant")
else()
add_custom_command(
    TARGET nnfusion_naive_rt
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/Constant ${CMAKE_BINARY_DIR}/Constant
)
endif()
)";
            auto host_async_manager =
                nnfusion::async::AsyncManagerFactory::get_host_async_manager(m_graph, GENERIC_CPU);
            if (host_async_manager->num_non_default_stream() > 0)
            {
                lu << "# need to specify the correct path of eigen\n"
                   << "set(EIGEN_DIR \"/usr/include/eigen3\" CACHE STRING \"EIGEN libraries folder "
                      "location\")\n"
                   << "include_directories(${EIGEN_DIR})\n\n";
                lu << "include(threadpool/threadpool.cmake)\n";
                lu << "target_link_libraries(nnfusion_naive_rt threadpool)\n\n";
            }
            return lu.get_code();
        }

        virtual void post_projgen(void) override
        {
            //generate CMakeList.txt
            this->lu_cmakefile = make_shared<LanguageUnit>("CMakeLists.txt");
            LanguageUnit& lu_cmake = *this->lu_cmakefile;
            lu_cmake << get_generate_cmakelists();
        }

        virtual void after_projgen(void) override
        {
            auto hipify_exec =
                nnfusion::codegen::get_file_from_templates("rocm_adapter/hipify-nnfusion");
            // update for nnfusion_rt.cu
            NNFUSION_CHECK(
                0 == system((hipify_exec +
                             " nnfusion_rt.cu | grep -v 'include.*cublas_v2' | grep -v "
                             "'include.*cuda.h' | grep -v 'include.*cudnn' > nnfusion_rt.cpp && rm "
                             "nnfusion_rt.cu")
                                .c_str()));

            // for rocm 3.5 compatibility purpose
            string rocm35cmd = "sed -i 's/extern *__shared__/__shared__/g' nnfusion_rt.cpp";
            NNFUSION_CHECK(0 == system((rocm35cmd).c_str())) << rocm35cmd;

            // update for main_test.cpp
            NNFUSION_CHECK(
                0 ==
                system("sed -i 's/^.*include.*cuda_profiler_api.*$//g' main_test.cpp && sed -i "
                       "'s/cudaProfiler.*\\(.*\\)//g' main_test.cpp"));
            // update for nnfusion_rt.h
            NNFUSION_CHECK(
                0 == system("sed -i 's/<cuda\\.h>/\"rocm_adapter.h\"/g' nnfusion_rt.h && sed -i "
                            "'s/cuda_runtime\\.h/hip\\/hip_runtime.h/g' nnfusion_rt.h"));

            // Update for kernels
            if (FLAGS_fkernels_as_files)
            {
                auto list_dir = [](string path, string has_str) {
                    vector<string> files;
                    struct dirent* entry;
                    DIR* dir = opendir(path.c_str());
                    if (dir == NULL)
                    {
                        return files;
                    }

                    while ((entry = readdir(dir)) != NULL)
                    {
                        string tstr(entry->d_name);
                        if (tstr.find(has_str) < tstr.length() && tstr.length() > 3)
                            files.push_back(path + "/" + tstr);
                    }
                    closedir(dir);
                    return files;
                };

                auto kernels = list_dir("kernels", ".cu");
                for (auto kernel : kernels)
                {
                    auto raw_kernel = kernel;
                    kernel.replace(kernel.find(".cu"), 3, ".cpp");
                    string grepv = hipify_exec + " " + raw_kernel +
                                   " | grep -v 'include.*cublas_v2' | grep -v 'include.*cuda.h' | "
                                   "grep -v 'include.*cudnn' > " +
                                   kernel + " && rm " + raw_kernel;
                    NNFUSION_CHECK(0 == system((grepv).c_str())) << grepv;
                    string hipcmd =
                        "sed -i 's/#include <hip\\/hip_runtime.h>/#include "
                        "\"..\\/rocm_adapter.h\"\\n#include "
                        "<hip\\/hip_runtime.h>/g' " +
                        kernel;
                    NNFUSION_CHECK(0 == system((hipcmd).c_str())) << hipcmd;
                    hipcmd = "sed -i 's/extern *__shared__/__shared__/g' " + kernel;
                    NNFUSION_CHECK(0 == system((hipcmd).c_str())) << hipcmd;
                }
            }
            // update for rocm_adapter.h
            nnfusion::codegen::copy_file_from_templates("rocm_adapter/rocm_adapter.h",
                                                        "./rocm_adapter.h");
            // fast compile script for dynamic shared lib
            nnfusion::codegen::copy_file_from_templates(
                "rocm_adapter/fastgen_for_sliced_kernels.sh", "./fastgen_for_sliced_kernels.sh");
            NNFUSION_CHECK(0 == system("chmod a+x fastgen_for_sliced_kernels.sh"));

            // create_image_tests
            nnfusion::codegen::copy_file_from_templates("image_tests/image_test.cpp",
                                                        "./image_tests/image_test.cpp");
            nnfusion::codegen::copy_file_from_templates("image_tests/CMakeLists_rocm.txt",
                                                        "./image_tests/CMakeLists.txt");

            //generate CMakeList.txt
            LanguageUnit& lu_cmake = *this->lu_cmakefile;
            lu_cmake << get_generate_cmakelists();

            if (super_scaler_enable)
            {
                nnfusion::codegen::copy_file_from_templates("super_scaler/super_scaler.h",
                                                            "./super_scaler.h");
                NNFUSION_LOG(NNFUSION_WARNING)
                    << "libsuper_scaler_rocm.so should be copied from "
                       "(build)/src/tools/nnfusion/templates/super_scaler/";
                nnfusion::codegen::copy_file_from_templates("super_scaler/libsuper_scaler_rocm.so",
                                                            "./libsuper_scaler_rocm.so");
            }
        }

        virtual KernelEmitter::Pointer
            match_kernel(std::vector<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>& res)
        {
            for (auto& k : res)
            {
                if (k.second != nullptr && k.first == device_type() &&
                    k.second->get_or_emit_source(true) != nullptr)
                {
                    return k.second;
                }
            }
            // if there is no valid ROCm kernel, use the CUDA kernel
            for (auto& k : res)
            {
                if (k.second != nullptr && k.first == NNFusion_DeviceType::CUDA_GPU &&
                    k.second->get_or_emit_source(true) != nullptr)
                {
                    return k.second;
                }
            }
            return nullptr;
        }

        virtual std::string get_target_name(void) override { return "rocm_codegen"; }
        virtual std::vector<shared_ptr<const KernelRegistration>>
            find_backend_kernels(const std::string& op_name,
                                 const shared_ptr<KernelContext>& ctx) override
        {
            auto kernel_regs =
                KernelRegistry::Global()->FindKernelRegistrations(op_name, ROCM_GPU, DT_FLOAT);
            for (auto it :
                 KernelRegistry::Global()->FindKernelRegistrations(op_name, CUDA_GPU, DT_FLOAT))
                kernel_regs.push_back(it);

            {
                auto priority = [](const std::string& tag) -> int {
                    static char sym_prio[] = "PRIORITY_";
                    int at = tag.find(sym_prio);
                    return (at != 0) ? 0 : atoi(tag.substr(sizeof(sym_prio) - 1).c_str());
                };

                std::sort(kernel_regs.begin(),
                          kernel_regs.end(),
                          [&](const shared_ptr<const KernelRegistration>& x,
                              const shared_ptr<const KernelRegistration>& y) {
                              auto x_prio = priority(x->m_tag), y_prio = priority(y->m_tag);
                              if (x_prio != y_prio)
                                  return x_prio > y_prio;

                              auto x_type = x->m_factory(ctx)->get_kernel_type();
                              auto y_type = y->m_factory(ctx)->get_kernel_type();
                              if (x_type != y_type)
                                  return x_type < y_type;

                              return false;
                          });
            }
            return std::move(kernel_regs);
        }
    };

    std::shared_ptr<IInterpreterPass> make_rocm_codegenerator()
    {
        return std::make_shared<RocmCodeGenerator>();
    }
} // namespace nnfusion