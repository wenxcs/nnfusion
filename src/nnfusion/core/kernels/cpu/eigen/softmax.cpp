// Microsoft (c) 2019, NNFusion Team

#include "softmax.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Softmax",                                                 // op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("eigen"), // attrs
                        cpu::SoftmaxEigen<float>)                                  // constructor
