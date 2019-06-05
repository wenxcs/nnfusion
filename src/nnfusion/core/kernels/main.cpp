// Microsoft (c) 2019, NNFusion Team

#include "kernel_registration.hpp"

int main()
{
    using namespace nnfusion;

    std::cout << "Global registered kernel size: "
              << kernels::KernelRegistry::Global()->RegisteredKernelSize() << std::endl;
    auto kernel_reg =
        kernels::KernelRegistry::Global()->FindKernelRegistration("Pad", CUDA_GPU, DT_FLOAT);
    if (kernel_reg)
    {
        std::cout << "Find registered kernel for < Pad, CUDA_GPU, DT_FLOAT> " << std::endl;
        kernel_reg->debug_string();
    }
    else
    {
        std::cout << "No registered kernel found for < Pad, CUDA_GPU, DT_FLOAT> " << std::endl;
    }

    kernel_reg =
        kernels::KernelRegistry::Global()->FindKernelRegistration("Pad", ROCM_GPU, DT_FLOAT);
    if (kernel_reg)
    {
        std::cout << "Find registered kernel for < Pad, ROCM_GPU, DT_FLOAT> " << std::endl;
    }
    else
    {
        std::cout << "No registered kernel found for < Pad, ROCM_GPU, DT_FLOAT> " << std::endl;
    }

    kernel_reg =
        kernels::KernelRegistry::Global()->FindKernelRegistration("Pad", CUDA_GPU, DT_INT32);
    if (kernel_reg)
    {
        std::cout << "Find registered kernel for < Pad, CUDA_GPU, DT_INT32> " << std::endl;
        kernel_reg->debug_string();
    }
    else
    {
        std::cout << "No registered kernel found for < Pad, CUDA_GPU, DT_INT32> " << std::endl;
    }

    return 0;
}