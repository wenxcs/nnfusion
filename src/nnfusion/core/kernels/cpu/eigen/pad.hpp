// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "../cpu_kernel_emitter.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            template <typename ElementType>
            class Pad : public CpuKernelEmitter
            {
            public:
                Pad(shared_ptr<KernelContext> ctx)
                    : CpuKernelEmitter(ctx)
                {
                    auto pad = static_pointer_cast<ngraph::op::Pad>(ctx->node);
                    input_shape = ngraph::Shape(ctx->inputs[0].get_shape());
                    output_shape = ngraph::Shape(ctx->outputs[0].get_shape());
                    padding_below = ngraph::Shape(pad->get_padding_below());
                    padding_above = ngraph::Shape(pad->get_padding_above());
                    padding_interior = ngraph::Shape(pad->get_padding_interior());
                    input_type = ctx->inputs[0].get_element_type().c_type_string();
                    output_type = ctx->outputs[0].get_element_type().c_type_string();

                    rank = static_cast<uint32_t>(input_shape.size());

                    std::stringstream tag;
                    tag << rank << "pad_i" << join(input_shape, "_") << "pad_o"
                        << join(output_shape, "_") << "_pb" << join(padding_below, "_") << "_pi"
                        << join(padding_interior, "_");
                    custom_tag = tag.str();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    // function signature:
                    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)

                    if (m_context->inputs[0].get_element_type() == element::f32 &&
                        padding_interior == Shape(input_shape.size()))
                    {
                        std::vector<string> padding;
                        for (int i = 0; i < rank; i++)
                        {
                            std::stringstream ss;
                            ss << "{" << padding_below[i] << ", " << padding_above[i] << "}";
                            padding.push_back(ss.str());
                        }

                        auto code = ngraph::op::create_code_from_template(
                            R"(
                                Eigen::array<Eigen::Index, @Rank@> out_dims(@out_dims@);
                                Eigen::array<Eigen::Index, @Rank@> in_dims(@in_dims@);
                                Eigen::array<Eigen::IndexPair<size_t>, @Rank@> padding(@padding@);

                                // for (int i = 0; i < @Rank@; i++)
                                // {
                                //     out_dims[i] = output_shape[i];
                                //     in_dims[i] = input_shape[i];
                                //     padding[i] = {padding_below[i], padding_above[i]};
                                // }
                                Eigen::TensorMap<Eigen::Tensor<@ElementType@, @Rank@, Eigen::RowMajor>> out(
                                    static_cast<@ElementType@*>(output0), out_dims);
                                Eigen::TensorMap<Eigen::Tensor<@ElementType@, @Rank@, Eigen::RowMajor>> in(
                                    static_cast<@ElementType@*>(input0), in_dims);

                                out.device(eigen::global_thread_pool_device) =
                                    in.pad(padding, *static_cast<@ElementType@*>(*input1));
                            )",
                            {{"Rank", rank},
                             {"out_dims", join(output_shape)},
                             {"in_dims", join(input_shape)},
                             {"padding", join(padding)},
                             {"ElementType", output_type}});
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
                    _lu->require(header::thread);
                    _lu->require(header::eigen_tensor);
                    _lu->require(declaration::eigen_global_thread_pool);
                    _lu->require(declaration::eigen_global_thread_pool_device);

                    return _lu;
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                ngraph::Shape input_shape, output_shape, padding_above, padding_below,
                    padding_interior;
                uint32_t rank;
                string input_type, output_type;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion