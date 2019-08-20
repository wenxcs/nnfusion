// Microsoft (c) 2019, Wenxiang Hu

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

#include "cuda_codegenerator.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/op.hpp"

namespace nnfusion
{
    class RocmCodeGenerator : public CudaCodeGenerator
    {
    public:
        DeviceType device_type() override { return DeviceType::ROCM_GPU; }
        virtual std::string get_generate_cmakelists(void) override
        {
            return R"(project(main_test)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
set(CMAKE_CXX_FLAGS "-O2 -Wno-ignored-attributes")

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
        }

        virtual void post_projgen(void) override
        {
            auto hipify_exec =
                nnfusion::codegen::get_file_from_templates("rocm_adapter/hipify-nnfusion");
            // update for nnfusion_rt.cu
            assert(0 ==
                   system((hipify_exec +
                           " nnfusion_rt.cu | grep -v 'include.*cublas_v2' | grep -v "
                           "'include.*cuda.h' | grep -v 'include.*cudnn' > nnfusion_rt.cpp && rm "
                           "nnfusion_rt.cu")
                              .c_str()));
            // update for main_test.cpp
            assert(0 ==
                   system("sed -i 's/^.*include.*cuda_profiler_api.*$//g' main_test.cpp && sed -i "
                          "'s/cudaProfiler.*\\(.*\\)//g' main_test.cpp"));
            // update for nnfusion_rt.h
            assert(0 == system("sed -i 's/<cuda\\.h>/\"rocm_adapter.h\"/g' nnfusion_rt.h && sed -i "
                               "'s/cuda_runtime\\.h/hip\\/hip_runtime.h/g' nnfusion_rt.h"));
            // update for rocm_adapter.h
            nnfusion::codegen::copy_file_from_templates("rocm_adapter/rocm_adapter.h",
                                                        "./rocm_adapter.h");

            // create_image_tests
            nnfusion::codegen::copy_file_from_templates("image_tests/image_test.cpp",
                                                        "./image_tests/image_test.cpp");
            nnfusion::codegen::copy_file_from_templates("image_tests/CMakeLists_rocm.txt",
                                                        "./image_tests/CMakeLists.txt");
        }

        virtual std::string get_target_name(void) override { return "rocm_codegen"; }
        virtual std::vector<shared_ptr<const KernelRegistration>>
            find_backend_kernels(const std::string& op_name,
                                 const shared_ptr<KernelContext>& ctx) override
        {
            auto kernel_regs =
                KernelRegistry::Global()->FindKernelRegistrations(op_name, ROCM_GPU, DT_FLOAT);
            if (!kernel_regs.size())
                kernel_regs =
                    KernelRegistry::Global()->FindKernelRegistrations(op_name, CUDA_GPU, DT_FLOAT);
            else
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
