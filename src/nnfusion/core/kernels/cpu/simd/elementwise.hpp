// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "../cpu_helper.hpp"
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
            class ElementwiseSimd : public SimdKernelEmitter
            {
            public:
                ElementwiseSimd(shared_ptr<KernelContext> ctx)
                    : SimdKernelEmitter(ctx)
                {
                    m_data_size = m_context->inputs.front()->size(false);
                    for (auto arg : ctx->inputs)
                    {
                        m_data_types.push_back(arg->get_element_type().c_type_string());
                    }
                    m_data_types.push_back(ctx->outputs[0]->get_element_type().c_type_string());
                }

                LanguageUnit_p emit_function_body() override
                {
                    if (CpuOpMap<T>::simd_op == nullptr)
                    {
                        return nullptr;
                    }

                    size_t remainder_count = m_data_size % m_simd_block_size;
                    size_t loop_count = m_data_size - remainder_count;

                    auto op = CpuOpMap<T>::simd_op;

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    if (CpuOpMap<T>::simd_math_kernel != nullptr)
                    {
                        auto math_kernel = get_simd_math_kernel(
                            op, CpuOpMap<T>::simd_math_kernel, m_data_size, m_data_types);
                        NNFUSION_CHECK_NOT_NULLPTR(math_kernel);
                        lu.require(math_kernel);
                    }
                    auto num_inputs = m_data_types.size() - 1;
                    NNFUSION_CHECK(num_inputs > 0)
                        << "At least one input and one output tesnor for elementwise-op.";

                    if (m_data_types[num_inputs] == "bool")
                    {
                        lu << "float tmp_buffer[" << m_simd_block_size << "];";
                    }
                    if (loop_count > 0)
                    {
                        lu << "for (size_t i = 0; i < " << loop_count
                           << "; i+=" << m_simd_block_size << ")\n";
                        lu << "{\n";
                        for (size_t i = 0; i < num_inputs; ++i)
                        {
                            lu << "__m256 in" << i << " = _mm256_loadu_ps(input" << i << " + i);\n";
                        }
                        lu << "__m256 out = " << op << "(";

                        for (size_t i = 0; i < num_inputs - 1; ++i)
                        {
                            lu << "in" << i << ", ";
                        }
                        lu << "in" << num_inputs - 1 << ");\n";
                        if (m_data_types[num_inputs] == "bool")
                        {
                            lu << "_mm256_storeu_ps(tmp_buffer, out);\n";
                            lu << "for (int j = 0; j < " << m_simd_block_size << "; ++j)\n{\n";
                            lu << "output0[i + j] = (bool)tmp_buffer[j];\n}\n";
                        }
                        else
                        {
                            lu << "_mm256_storeu_ps(output0 + i, out);\n";
                        }
                        lu << "}\n";
                    }

                    if (remainder_count > 0)
                    {
                        lu << "for (size_t i = " << loop_count << "; i < " << m_data_size
                           << "; ++i)\n";
                        lu << "{\n";
                        for (size_t i = 0; i < num_inputs; ++i)
                        {
                            lu << "__m256 in" << i
                               << " = _mm256_insertf128_ps(_mm256_setzero_ps(), _mm_set_ss(input"
                               << i << "[i]), 0);\n";
                        }
                        lu << "__m256 out = " << op << "(";

                        for (size_t i = 0; i < num_inputs - 1; ++i)
                        {
                            lu << "in" << i << ", ";
                        }
                        lu << "in" << num_inputs - 1 << ");\n";
                        if (m_data_types[num_inputs] == "bool")
                            lu << "output0[i] = (bool)_mm_cvtss_f32(_mm256_extractf128_ps(out, "
                                  "0));\n";
                        else
                            lu << "output0[i] = _mm_cvtss_f32(_mm256_extractf128_ps(out, 0));\n";
                        lu << "}\n";
                    }

                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

                    return _lu;
                }

                virtual std::pair<std::string, shared_ptr<LanguageUnit>> get_op_kernel() override
                {
                    std::string op = CpuOpMap<T>::simd_op;
                    shared_ptr<LanguageUnit> kernel = nullptr;

                    if (CpuOpMap<T>::simd_math_kernel != nullptr)
                    {
                        kernel = get_simd_math_kernel(
                            op, CpuOpMap<T>::simd_math_kernel, m_data_size, m_data_types);
                        NNFUSION_CHECK_NOT_NULLPTR(kernel);
                    }
                    return std::make_pair(op, kernel);
                }

            protected:
                size_t m_data_size;
                vector<string> m_data_types;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
