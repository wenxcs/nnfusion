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
            class ElementwiseEigen : public EigenKernelEmitter
            {
            public:
                ElementwiseEigen(shared_ptr<KernelContext> ctx)
                    : EigenKernelEmitter(ctx)
                {
                    data_size = m_context->inputs.front()->size(false);
                    for (auto arg : ctx->inputs)
                    {
                        data_types.push_back(arg->get_element_type().c_type_string());
                    }
                    data_types.push_back(ctx->outputs[0]->get_element_type().c_type_string());
                }

                LanguageUnit_p emit_function_body() override
                {
                    if (CpuOpMap<T>::eigen_op == nullptr)
                    {
                        return nullptr;
                    }

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    if (m_context->inputs.size() == 1)
                    {
                        std::string op = CpuOpMap<T>::eigen_op;
                        auto code = nnfusion::op::create_code_from_template(
                            R"(
Eigen::Map<Eigen::Array<@in_type@,@data_size@,1> > in(&input0[0]);
Eigen::Map<Eigen::Array<@out_type@,@data_size@,1> > out(&output0[0]);
out = in.@op@();
)",
                            {{"in_type", data_types[0]},
                             {"out_type", data_types[1]},
                             {"data_size", data_size},
                             {"op", op}});
                        lu << code;
                    }
                    else if (m_context->inputs.size() == 2)
                    {
                        std::string op = CpuOpMap<T>::eigen_op;
                        auto code = nnfusion::op::create_code_from_template(
                            R"(
Eigen::Map<Eigen::Array<@in1_type@,@data_size@,1> > in1(&input0[0]);
Eigen::Map<Eigen::Array<@in2_type@,@data_size@,1> > in2(&input1[0]);
Eigen::Map<Eigen::Array<@out_type@,@data_size@,1> > out(&output0[0]);
out = in1 @op@ in2;
)",
                            {{"in1_type", data_types[0]},
                             {"in2_type", data_types[1]},
                             {"out_type", data_types[2]},
                             {"data_size", data_size},
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
                size_t data_size;
                vector<string> data_types;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
