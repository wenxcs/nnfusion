// Microsoft (c) 2019, NNFusion Team

#include "reduce.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_EW_KERNEL(OP_NAME)                                                                \
    REGISTER_KERNEL_EMITTER("" #OP_NAME "",                                                        \
                            Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("eigen"),             \
                            cpu::ReduceEigen<nnfusion::op::OP_NAME>);

REGISTER_EW_KERNEL(Sum)
REGISTER_EW_KERNEL(Product)
REGISTER_EW_KERNEL(Max)
REGISTER_EW_KERNEL(Min)
