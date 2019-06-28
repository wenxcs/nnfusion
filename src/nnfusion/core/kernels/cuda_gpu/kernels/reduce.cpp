// Microsoft (c) 2019, NNFusion Team

#include "reduce.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_GPU_KERNEL(OP_NAME)                                                               \
    REGISTER_KERNEL_EMITTER("Reduce",                                                              \
                            Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel" #OP_NAME), \
                            cuda::Reduce<ngraph::op::OP_NAME>)

REGISTER_GPU_KERNEL(Max)
REGISTER_GPU_KERNEL(Min)
