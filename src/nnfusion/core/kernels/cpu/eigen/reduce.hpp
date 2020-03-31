// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "../cpu_kernel_emitter.hpp"
#include "../cpu_kernelops.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            template <typename T>
            class ReduceEigen : public EigenKernelEmitter
            {
            public:
                ReduceEigen(shared_ptr<KernelContext> ctx)
                    : EigenKernelEmitter(ctx)
                {
                    auto op = static_pointer_cast<T>(ctx->gnode->get_op_ptr());
                    NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not expected.";

                    reduce_axis = op->get_reduction_axes();
                    input_shape = ctx->inputs[0]->get_shape();
                    output_size = m_context->outputs.front()->size(false);
                    data_type = ctx->outputs[0]->get_element_type().c_type_string();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    // Handle the cases that input tensor is matrix.
                    if (CpuOpMap<T>::eigen_op != nullptr && input_shape.size() == 2)
                    {
                        for (auto axis : reduce_axis)
                        {
                            NNFUSION_CHECK(axis == 0 || axis == 1)
                                << "Reduce axis is not expected for matrix.";
                        }

                        std::string op = CpuOpMap<T>::eigen_op;
                        auto code = nnfusion::op::create_code_from_template(
                            R"(
Eigen::Map<Eigen::Matrix<@data_type@,@row@,@col@>> in(&input0[0]);
Eigen::Map<Eigen::Array<@data_type@,@output_size@,1> > out(&output0[0]);
out = in.@axis@().@op@();
)",
                            {{"data_type", data_type},
                             {"row", input_shape[0]},
                             {"col", input_shape[1]},
                             {"output_size", output_size},
                             {"axis", *(reduce_axis.begin()) == 1 ? "rowwise" : "colwise"},
                             {"op", op}});
                        lu << code;
                    }
                    else
                    {
                        return nullptr;
                    }

                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::eigen_tensor);

                    return _lu;
                }

            private:
                size_t output_size;
                string data_type;
                nnfusion::Shape input_shape;
                nnfusion::AxisSet reduce_axis;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
