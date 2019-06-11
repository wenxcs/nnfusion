// Microsoft (c) 2019, Wenxiang Hu
#include "codegenerator.hpp"

#include "nnfusion/core/kernels/kernel_registration.hpp"

using namespace nnfusion;

bool CodeGenerator::run(std::shared_ptr<InterpreterContext> ctx,
                        std::shared_ptr<TranslationUnit> tu)
{
    auto kernel_reg =
        kernels::KernelRegistry::Global()->FindKernelRegistration("Pad", CUDA_GPU, DT_FLOAT);
    return true;
}