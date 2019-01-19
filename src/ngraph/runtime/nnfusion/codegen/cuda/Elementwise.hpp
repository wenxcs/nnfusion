// Microsoft (c) 2019, Wenxiang
#pragma once
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_helper.hpp"
#include "ngraph/runtime/nnfusion/intermediate/op_tbl.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_op.hpp"

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
                    class Elementwise : public CodeGenOP
                    {
                    private:
                        shared_ptr<intermediate::Elementwise<T>> inter_op;

                    public:
                        Elementwise(shared_ptr<IntermediateOP> inter_op)
                            : CodeGenOP(inter_op)
                        {
                            assert_nullptr(
                                this->inter_op =
                                    static_pointer_cast<intermediate::Elementwise<T>>(inter_op));
                        }

                        string codegen_source_name() override
                        {
                            return codegen_function_name() + ".cu";
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

                        shared_ptr<CodeWriter> codegen_function_definition() override
                        {
                            shared_ptr<CodeWriter> cw(new CodeWriter);
                            CodeWriter& writer = *cw;
                            std::string name = codegen_function_name();
                            std::string op = CudaOpMap<T>::op;
                            std::vector<std::string>& data_types = inter_op->dtypes;

                            get_math_kernel(writer, op, CudaOpMap<T>::math_kernel, data_types);

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

                        shared_ptr<CodeWriter> codegen_function_call() override
                        {
                            shared_ptr<CodeWriter> cw(new CodeWriter);
                            CodeWriter& writer = *cw;

                            uint32_t nthreads = static_cast<uint32_t>(ngraph::shape_size(inter_op->out[0].get_shape()));
                            // TODO: currently we set it to 64, will add tuning method later
                            uint32_t block_size_x = 512;
                            int num_SMs;
                            CUDA_RT_SAFE_CALL(cudaDeviceGetAttribute(
                                &num_SMs, cudaDevAttrMultiProcessorCount, 0));
                            uint32_t aligned_grid_size_x =
                                fmin(num_SMs * 32, align_to_block_size(nthreads, block_size_x));

                            writer << codegen_function_name() << "<<<("
                                   << aligned_grid_size_x << ", "
                                   << 1 << ", "
                                   << 1 << "), ("
                                   << block_size_x << ", "
                                   << 1 << ", "
                                   << 1 << "), "
                                   << 0 << ", "
                                   << 0 << ">>>"
                                   << "("
                                   << join(inter_op->arg_names, ", ")
                                   << ", "
                                   << join(inter_op->out_names, ", ")
                                   << ", "
                                   << nthreads
                                   << ");\n";

                            return cw;
                        }

                        //This is for Relu
                        shared_ptr<CodeWriter> codegen_test() override
                        {
                            shared_ptr<CodeWriter> cw(new CodeWriter);
                            CodeWriter& writer = *cw;
                            writer << "// Relu Test\n";
                            vector<float> data;
                            // Malloc
                            for(auto& arg: inter_op->args)
                            {
                                data = test_hostData(writer, arg);
                                test_cudaMalloc(writer, arg);
                                test_cudaMemcpyHtoD(writer, arg);
                            }

                            for(int i=0; i<data.size();i++)
                            {
                                if(data[i]<0) data[i] = 0;
                            }
                            test_hostData(writer, inter_op->out[0], data);
                            test_cudaMalloc(writer, inter_op->out[0]);
                            writer << codegen_function_call()->get_code();
                            test_cudaMemcpyDtoH(writer, inter_op->out[0]);
                            test_compare(writer, inter_op->out[0]);
                            writer << "printf(\"SUCCEED\\n\");";
                            return cw;
                        }

                        shared_ptr<CodeWriter> codegen_dependency() override
                        {
                            shared_ptr<CodeWriter> cw(new CodeWriter);
                            CodeWriter& writer = *cw;
                            writer << "#include <cuda.h>\n";
                            writer << "#include <stdio.h>\n";
                            return cw;
                        }

                    public:
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