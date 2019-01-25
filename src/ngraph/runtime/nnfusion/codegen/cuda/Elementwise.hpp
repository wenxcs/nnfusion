// Microsoft (c) 2019, Wenxiang
#pragma once
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_codegenop.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_helper.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_langunit.hpp"
#include "ngraph/runtime/nnfusion/intermediate/op_tbl.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            namespace codegen
            {
                namespace cuda
                {
                    template <class T>
                    class Elementwise : public CudaCodeGenOP
                    {
                    private:
                        shared_ptr<intermediate::Elementwise<T>> inter_op;

                    public:
                        Elementwise(shared_ptr<IntermediateOP> inter_op)
                            : CudaCodeGenOP(inter_op)
                        {
                            assert_nullptr(
                                this->inter_op =
                                    static_pointer_cast<intermediate::Elementwise<T>>(inter_op));
                        }

                        string codegen_function_name() override
                        {
                            // kernel_name is used to check if the cuda kernel has been previously compiled
                            std::stringstream kernel_name;
                            kernel_name << "cuda"
                                        << "_ew"
                                        << "_" << CudaOpMap<T>::op << "_"
                                        << join(inter_op->dtypes, "_");

                            return kernel_name.str();
                        }

                        string codegen_test_name() override
                        {
                            return codegen_function_name() + "_test";
                        }

                        shared_ptr<LanguageUnit> codegen_function_definition() override
                        {
                            std::string name = codegen_function_name();
                            shared_ptr<LanguageUnit> cw(new LanguageUnit(name));
                            LanguageUnit& writer = *cw;
                            std::string op = CudaOpMap<T>::op;
                            std::vector<std::string>& data_types = inter_op->dtypes;

                            auto math_kernel =
                                get_math_kernel(op, CudaOpMap<T>::math_kernel, data_types);
                            assert_nullptr(math_kernel);
                            cw->require(math_kernel);

                            auto num_inputs = data_types.size() - 1;
                            assert_bool(num_inputs > 0)
                                << "At least one input and one output tesnor for elementwise-op.";
                            writer << "extern \"C\" __global__ void " << name << "(";
                            for (size_t i = 0; i < num_inputs; i++)
                            {
                                writer << data_types[i] << "* in" << i << ", ";
                            }
                            writer << data_types[num_inputs] << "* out, "
                                   << "uint32_t n)\n";
                            writer.block_begin();
                            {
                                writer
                                    << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; \n";
                                writer << "uint32_t step = gridDim.x * blockDim.x; \n";
                                writer << "for ( ;tid < n; tid += step)\n";
                                writer.block_begin();
                                {
                                    writer << "out[tid] = " << op << "(";
                                    for (size_t i = 0; i < num_inputs - 1; i++)
                                    {
                                        writer << "in" << i << "[tid], ";
                                    }
                                    writer << "in" << num_inputs - 1 << "[tid]);\n";
                                }
                                writer.block_end();
                            }
                            writer.block_end();

                            return cw;
                        }

                        shared_ptr<LanguageUnit> codegen_function_call() override
                        {
                            std::string name = codegen_function_name() + "_call";
                            shared_ptr<LanguageUnit> cw(new LanguageUnit(name));
                            LanguageUnit& writer = *cw;

                            uint32_t nthreads = static_cast<uint32_t>(
                                ngraph::shape_size(inter_op->out[0].get_shape()));
                            // TODO: currently we set it to 64, will add tuning method later
                            uint32_t block_size_x = 512;
                            int num_SMs;
                            CUDA_RT_SAFE_CALL(cudaDeviceGetAttribute(
                                &num_SMs, cudaDevAttrMultiProcessorCount, 0));
                            uint32_t aligned_grid_size_x =
                                fmin(num_SMs * 32, align_to_block_size(nthreads, block_size_x));

                            writer << codegen_function_name() << "<<<(" << aligned_grid_size_x
                                   << ", " << 1 << ", " << 1 << "), (" << block_size_x << ", " << 1
                                   << ", " << 1 << "), " << 0 << ", " << 0 << ">>>"
                                   << "(" << join(inter_op->arg_names, ", ") << ", "
                                   << join(inter_op->out_names, ", ") << ", " << nthreads << ");\n";

                            return cw;
                        }

                        shared_ptr<LanguageUnit> codegen_dependency() override
                        {
                            std::string name = codegen_function_name() + "_dep";
                            shared_ptr<LanguageUnit> cw(new LanguageUnit(name));

                            cw->require(header::cuda);
                            cw->require(header::stdio);

                            return cw;
                        }

                        static std::shared_ptr<CodeGenOP>
                            codegen(std::shared_ptr<IntermediateOP> inter_op)
                        {
                            shared_ptr<Elementwise> cop(new Elementwise(inter_op));
                            NGRAPH_DEBUG << "Codegen for Elementwise function:"
                                         << cop->codegen_function_name() << endl;
                            return cop;
                        }
                    };
                }
            }
        }
    }
}