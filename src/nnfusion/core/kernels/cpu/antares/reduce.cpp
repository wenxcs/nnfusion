// Microsoft (c) 2019, NNFusion Team

#include "reduce.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_REDUCE_KERNEL(OP_NAME)                                                            \
    REGISTER_KERNEL_EMITTER("" #OP_NAME "",                                                        \
                            Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"),           \
                            cpu::ReduceAntares<nnfusion::op::OP_NAME>);

REGISTER_REDUCE_KERNEL(Sum)
REGISTER_REDUCE_KERNEL(Product)
REGISTER_REDUCE_KERNEL(Max)
REGISTER_REDUCE_KERNEL(Min)
