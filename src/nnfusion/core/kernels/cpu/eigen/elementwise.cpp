// Microsoft (c) 2019, NNFusion Team

#include "elementwise.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_EW_KERNEL(OP_NAME)                                                                \
    REGISTER_KERNEL_EMITTER("" #OP_NAME "",                                                        \
                            Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("eigen"),             \
                            cpu::ElementwiseEigen<nnfusion::op::OP_NAME>);

REGISTER_EW_KERNEL(Rsqrt)
REGISTER_EW_KERNEL(Sqrt)
REGISTER_EW_KERNEL(Square)
REGISTER_EW_KERNEL(Add)
REGISTER_EW_KERNEL(Subtract)
REGISTER_EW_KERNEL(Multiply)
REGISTER_EW_KERNEL(Divide)
