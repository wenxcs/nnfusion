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
                    auto _op =
                        static_pointer_cast<nnfusion::op::Broadcast>(ctx->gnode->get_op_ptr());
                    CHECK_NOT_NULLPTR(_op) << "Node type is not Broadcast.";

                    nnfusion::Shape input_shape = ctx->inputs[0]->get_shape();
                    nnfusion::Shape output_shape = ctx->outputs[0]->get_shape();

                    auto expression = op::create_code_from_template(
                        R"(- input("input0", @input_shape@); output(@output_shape@, topi=topi.broadcast_to(args("input0"), @output_shape@));
)",
                        {{"input_shape", vector_to_string(input_shape)},
                         {"output_shape", vector_to_string(output_shape)}});

                    initialize(expression);
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
