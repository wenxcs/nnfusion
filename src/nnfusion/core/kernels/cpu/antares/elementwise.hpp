// Microsoft (c) 2019, NNFusion Team
#pragma once

#include <sstream>
#include <string>
#include <vector>
#include "../cpu_kernel_emitter.hpp"
#include "antares_ops.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            template <class T>
            class AntaresElementwise : public AntaresCpuKernelEmitter
            {
            public:
                AntaresElementwise(shared_ptr<KernelContext> ctx)
                    : AntaresCpuKernelEmitter(ctx)
                {
                    nnfusion::Shape shape = ctx->inputs[0]->get_shape();

                    std::stringstream expression;
                    expression << "- ";
                    size_t num_inputs = ctx->inputs.size();
                    for (size_t i = 0; i < num_inputs; ++i)
                    {
                        expression << "input(\"input" << i << "\", "
                                   << vector_to_string(shape, " * ") << "); ";
                    }
                    std::string tvm_op = TvmOpMap<T>::op;
                    expression << "output(" << vector_to_string(shape, " * ") << ", topi=" << tvm_op
                               << "(";
                    for (size_t i = 0; i < num_inputs - 1; ++i)
                    {
                        expression << "args(\"input" << i << "\"), ";
                    }
                    expression << "args(\"input" << (num_inputs - 1) << "\")));";

                    initialize(expression.str());
                }
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
