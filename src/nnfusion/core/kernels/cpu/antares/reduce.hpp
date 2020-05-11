// Microsoft (c) 2019, NNFusion Team
#pragma once

#include <sstream>
#include <string>
#include "../cpu_kernel_emitter.hpp"
#include "../cpu_kernelops.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            template <class T>
            class ReduceAntares : public AntaresCpuKernelEmitter
            {
            public:
                ReduceAntares(shared_ptr<KernelContext> ctx)
                    : AntaresCpuKernelEmitter(ctx)
                {
                    auto op = static_pointer_cast<T>(ctx->gnode->get_op_ptr());
                    NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not expected.";
                    nnfusion::AxisSet reduce_axis = op->get_reduction_axes();

                    nnfusion::Shape input_shape = ctx->inputs[0]->get_shape();
                    nnfusion::Shape output_shape = ctx->outputs[0]->get_shape();

                    // Handle the cases that input tensor is not matrix.
                    std::string tvm_op = CpuOpMap<T>::antares_op;

                    auto expression = op::create_code_from_template(
                        R"( - input("input0", @input_shape@); output(@output_shape@, topi=@tvm_op@(args("input0"), axis=@axis@, keepdims=True)); )",
                        {{"input_shape", vector_to_string(input_shape)},
                         {"output_shape", vector_to_string(output_shape)},
                         {"tvm_op", tvm_op},
                         {"axis", vector_to_string(reduce_axis)}});

                    initialize(expression);
                }
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
