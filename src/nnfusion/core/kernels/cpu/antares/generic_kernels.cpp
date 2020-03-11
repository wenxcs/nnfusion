// Microsoft (c) 2019, NNFusion Team

#include <iterator>
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
            class GenericAntares : public AntaresCpuKernelEmitter
            {
            public:
                GenericAntares(shared_ptr<KernelContext> ctx)
                    : AntaresCpuKernelEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    initialize(generic_op->m_expression);
                }

            private:
                shared_ptr<nnfusion::op::GenericOp> generic_op;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Tile",                                                      //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"), //attrs
                        cpu::GenericAntares)                                         //constructor

REGISTER_KERNEL_EMITTER("GatherV2",                                                  //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"), //attrs
                        cpu::GenericAntares)                                         //constructor

REGISTER_KERNEL_EMITTER("OneHot",                                                    //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"), //attrs
                        cpu::GenericAntares)                                         //constructor

REGISTER_KERNEL_EMITTER("BatchMatMul",                                               //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("antares"), //attrs
                        cpu::GenericAntares)                                         //constructor
