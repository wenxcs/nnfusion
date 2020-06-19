// Microsoft (c) 2019, NNFusion Team

#include "../cpu_kernel_emitter.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_ANTARES_KERNEL(OP_NAME)                                                           \
    REGISTER_KERNEL_EMITTER(#OP_NAME,                                                              \
                            Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"),           \
                            cpu::AntaresCpuKernelEmitter)

REGISTER_ANTARES_KERNEL(Tile)
REGISTER_ANTARES_KERNEL(GatherV2)
REGISTER_ANTARES_KERNEL(OneHot)
REGISTER_ANTARES_KERNEL(BatchMatMul)
REGISTER_ANTARES_KERNEL(Broadcast)
REGISTER_ANTARES_KERNEL(Convert)
REGISTER_ANTARES_KERNEL(Reshape)
REGISTER_ANTARES_KERNEL(Reverse)
REGISTER_ANTARES_KERNEL(Slice)
REGISTER_ANTARES_KERNEL(Concat)
REGISTER_ANTARES_KERNEL(Shape)
REGISTER_ANTARES_KERNEL(Pad)
REGISTER_ANTARES_KERNEL(Sum)

#undef REGISTER_ANTARES_KERNEL
