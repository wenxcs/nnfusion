// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            template <class T>
            class ElementWise : public CudaEmitter
            {
            public:
                ElementWise(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                {
                    enforce(ctx->outputs.size() == 1)
                        << "Multi-output elementwise ops are not currently supported.";

                    for (auto& arg : ctx->inputs)
                    {
                        data_types.push_back(arg.get_type());
                    }
                    data_types.push_back(ctx->outputs[0].get_type());
                }

                LanguageUnit_p emit_function_body() override
                {
                    create_ptr(LanguageUnit, lu_, get_function_name());
                    LanguageUnit& lu = *lu_;

                    std::string op = CudaOpMap<T>::op;

                    if (CudaOpMap<T>::math_kernel != nullptr)
                    {
                        auto math_kernel =
                            get_math_kernel(op, CudaOpMap<T>::math_kernel, data_types);
                        enforce_not_nullptr(math_kernel);
                        lu.require(math_kernel);
                    }

                    auto num_inputs = data_types.size() - 1;
                    uint32_t nthreads = static_cast<uint32_t>(
                        ngraph::shape_size(m_context->outputs[0].get_shape()));
                    enforce(num_inputs > 0)
                        << "At least one input and one output tesnor for elementwise-op.";

                    {
                        lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; \n";
                        lu << "uint32_t step = gridDim.x * blockDim.x; \n";
                        lu << "for ( ;tid < " << nthreads << "; tid += step)\n";
                        lu.block_begin();
                        {
                            lu << "output0[tid] = " << op << "(";
                            for (size_t i = 0; i < num_inputs - 1; i++)
                            {
                                lu << "input" << i << "[tid], ";
                            }
                            lu << "input" << num_inputs - 1 << "[tid]);\n";
                        }
                        lu.block_end();
                    }
                    return lu_;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(header::stdio);

                    return _lu;
                }

                void set_launch_config() override
                {
                    uint32_t nthreads = static_cast<uint32_t>(
                        ngraph::shape_size(m_context->outputs[0].get_shape()));
                    // TODO: currently we set it to 512, will add tuning method later
                    uint32_t block_size_x = 512;

                    m_gridDim = dim3(align_to_block_size(nthreads, block_size_x), 1, 1);
                    m_blockDim = dim3(block_size_x, 1, 1);
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                vector<string> data_types;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion