// Microsoft (c) 2019, NNFusion Team

#include <sstream>
#include <string>
#include <vector>
#include "../cpu_kernel_emitter.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class BroadcastAntares : public AntaresCpuKernelEmitter
            {
            public:
                BroadcastAntares(shared_ptr<KernelContext> ctx)
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
                        cpu::BroadcastAntares)                                       // constructor
