// Microsoft (c) 2019, NNFusion Team

#include "../cpu_kernel_emitter.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class GenericV1Antares : public AntaresCpuKernelEmitter
            {
            public:
                GenericV1Antares(shared_ptr<KernelContext> ctx)
                    : AntaresCpuKernelEmitter(ctx)
                {
                    initialize(nnfusion::op::get_translation(ctx->gnode));
                }
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("Broadcast",                                                 //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"), //attrs
                        cpu::GenericV1Antares)                                       //constructor

REGISTER_KERNEL_EMITTER("Convert",                                                   //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"), //attrs
                        cpu::GenericV1Antares)                                       //constructor

REGISTER_KERNEL_EMITTER("Reshape",                                                   //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"), //attrs
                        cpu::GenericV1Antares)                                       //constructor

REGISTER_KERNEL_EMITTER("Reverse",                                                   //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"), //attrs
                        cpu::GenericV1Antares)                                       //constructor

REGISTER_KERNEL_EMITTER("Slice",                                                     //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"), //attrs
                        cpu::GenericV1Antares)                                       //constructor

REGISTER_KERNEL_EMITTER("Concat",                                                    //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"), //attrs
                        cpu::GenericV1Antares)                                       //constructor

REGISTER_KERNEL_EMITTER("Shape",                                                     //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"), //attrs
                        cpu::GenericV1Antares)                                       //constructor

REGISTER_KERNEL_EMITTER("Pad",                                                       //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"), //attrs
                        cpu::GenericV1Antares)                                       //constructor
