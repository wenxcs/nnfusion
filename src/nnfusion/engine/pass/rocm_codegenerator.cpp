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
        virtual std::string get_generate_cmakelists(void) override
        {
            return R"(project(main_test)
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
        }

        virtual void post_projgen(void) override
        {
            // hipify kernel codes
            char exepath[1024];
            assert(readlink("/proc/self/exe", exepath, sizeof(exepath)) > 0);
            for (int i = strlen(exepath) - 1; i >= 0; --i)
                if (exepath[i] == '/')
                {
                    exepath[i] = 0;
                    break;
                }
            assert(
                0 ==
                system((std::string(exepath) +
                        "/hipify-nnfusion nnfusion_rt.cu | grep -v 'include.*cublas_v2' | grep -v "
                        "'include.*cuda.h' | grep -v 'include.*cudnn' > nnfusion_rt.cpp && rm "
                        "nnfusion_rt.cu")
                           .c_str()));
            assert(0 ==
                   system("sed -i 's/^.*include.*cuda_profiler_api.*$//g' main_test.cpp && sed -i "
                          "'s/cudaProfiler.*\\(.*\\)//g' main_test.cpp"));
            assert(0 == system("sed -i 's/<cuda\\.h>/\"rocm_adapter.h\"/g' nnfusion_rt.h && sed -i "
                               "'s/cuda_runtime\\.h/hip\\/hip_runtime.h/g' nnfusion_rt.h"));
            assert(0 == system((std::string("cp ") + exepath + "/hipify-adapter ./rocm_adapter.h")
                                   .c_str()));
        }

        virtual std::string get_target_name(void) override { return "rocm_codegen"; }
        virtual std::vector<shared_ptr<const KernelRegistration>>
            find_backend_kernels(const std::string& op_name) override
        {
            auto kernel_regs =
                KernelRegistry::Global()->FindKernelRegistrations(op_name, ROCM_GPU, DT_FLOAT);
            if (!kernel_regs.size())
                kernel_regs =
                    KernelRegistry::Global()->FindKernelRegistrations(op_name, CUDA_GPU, DT_FLOAT);
            return std::move(kernel_regs);
        }
    };

    std::shared_ptr<IInterpreterPass> make_rocm_codegenerator()
    {
        return std::make_shared<RocmCodeGenerator>();
    }
} // namespace nnfusion
