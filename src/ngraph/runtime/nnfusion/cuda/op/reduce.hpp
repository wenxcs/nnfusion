// Microsoft (c) 2019, Wenxiang
#pragma once
#include "../cuda_function.hpp"
#include "../cuda_helper.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace cuda
    {
        template <class T>
        class Reduce : public CudaFunction
        {
        private:
            ir::Reduce_p op;
            size_t data_bytes, rank, reduce_rank, out_rank;
            NVShape input_shape;
            uint32_t block_size_x_acc, nthreads_acc;
            string reduce_op;

            NVShape output_shape;
            NVShape non_reduce_strides;
            NVShape reduce_shape;
            NVShape reduce_strides;
            NVShape input_strides;

        public:
            Reduce(ir::Operator_p inter_op)
                : CudaFunction(inter_op)
            {
                assert_nullptr(this->op = static_pointer_cast<ir::Reduce>(inter_op));

                input_shape = op->args[0].get_shape();
                data_bytes = op->out[0].get_element_type().size();
                rank = input_shape.size();
                reduce_rank = op->reduce_axis.size();
                out_rank = rank - reduce_rank;
                // int num_SMs;
                block_size_x_acc = 256;
                // nthreads_acc = num_SMs * block_size_x_acc;

                reduce_op = CudaOpMap<T>::op;
            }

            string codegen_function_name() override
            {
                // kernel_name is used to check if the cuda kernel has been previously compiled
                std::stringstream kernel_name;
                kernel_name << "cuda"
                            << "_reduce"
                            << "_" << CudaOpMap<T>::op << "_" << join(op->dtypes, "_") << "_s_"
                            << join(input_shape, "_") << "_axis_" << join(op->reduce_axis, "_");

                return kernel_name.str();
            }

            string codegen_test_name() override { return codegen_function_name() + "_test"; }
            LanguageUnit_p codegen_function_definition_nd()
            {
                create_ptr(LanguageUnit, cw, codegen_function_name());
                LanguageUnit& writer = *cw;

                NVShape reduce_flag(rank, 0);
                for (auto a : op->reduce_axis)
                {
                    reduce_flag[a] = 1;
                }
                input_strides = row_major_strides(input_shape);
                for (int i = 0; i < rank; i++)
                {
                    if (reduce_flag[i] != 0)
                    {
                        reduce_shape.push_back(input_shape[i]);
                        reduce_strides.push_back(input_strides[i]);
                    }
                    else
                    {
                        non_reduce_strides.push_back(input_strides[i]);
                        output_shape.push_back(input_shape[i]);
                    }
                }
                NVShape output_strides = row_major_strides(output_shape);
                uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
                // TODO: currently we set it to 64, will add tuning method later
                uint32_t block_size_x = 64;
                uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

                writer << "extern \"C\" __global__ void " << writer.symbol << "_nd("
                       << op->dtypes[0] << "* in, " << op->dtypes[1] << "* out, size_t nthreads)\n";
                writer.block_begin();
                {
                    auto expand_vector_uint32 = [](string name, vector<uint32_t>& d) {
                        stringstream ss;
                        for (int i = 0; i < d.size(); i++)
                            ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
                        return ss.str();
                    };

                    writer << expand_vector_uint32("out_strides", output_strides);
                    writer << expand_vector_uint32("non_reduce_strides", non_reduce_strides);
                    writer << expand_vector_uint32("reduce_shape", reduce_shape);
                    writer << expand_vector_uint32("reduce_strides", reduce_strides);

                    writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; \n";
                    writer << "if (tid < nthreads)\n";
                    writer.block_begin();
                    {
                        if (out_rank > 0)
                        {
                            writer << "uint32_t dim_idx_generator = tid;\n";
                        }
                        writer << "uint32_t in_idx = 0;\n";
                        writer << op->dtypes[1] << " r = 0;\n";

                        // loop through all reduction axis
                        for (int64_t i = 0; i < static_cast<int64_t>(out_rank); i++)
                        {
                            writer << "in_idx += (dim_idx_generator / out_strides" << i
                                   << ") * non_reduce_strides" << i << ";\n";
                            writer << "dim_idx_generator %= out_strides" << i << ";\n";
                        }
                        int64_t last_r_idx = static_cast<int64_t>(reduce_rank) - 1;
                        for (int64_t j = 0; j < last_r_idx; j++)
                        {
                            writer << "for(int idx" << j << " = 0; idx" << j << "< reduce_shape"
                                   << j << "; idx" << j << "++)\n";
                            writer.block_begin();
                        }
                        {
                            writer << "uint32_t reduce_idx = in_idx;\n";
                            for (int64_t j = 0; j < last_r_idx; j++)
                            {
                                writer << "reduce_idx += idx" << j << " * reduce_strides" << j
                                       << ";\n";
                            }
                            writer << "int idx" << last_r_idx << " = 0;\n";
                            writer << "uint32_t step = reduce_strides" << last_r_idx << ";\n";
                            // unroll last reduction axis
                            uint32_t unroll_num = 8;
                            uint32_t unroll_shift = 3;
                            writer << "for(; idx" << last_r_idx << " < (reduce_shape" << last_r_idx
                                   << " >> " << unroll_shift << "); idx" << last_r_idx << "++)\n";
                            writer.block_begin();
                            {
                                for (int k = 0; k < unroll_num; k++)
                                {
                                    writer << "r = " << reduce_op << "(r , in[reduce_idx]);\n";
                                    writer << "reduce_idx += step;\n";
                                }
                            }
                            writer.block_end();
                            writer << "idx" << last_r_idx << " <<= " << unroll_shift << ";\n";
                            writer << "for(; idx" << last_r_idx << " < reduce_shape" << last_r_idx
                                   << "; idx" << last_r_idx << "++)\n";
                            writer.block_begin();
                            {
                                writer << "r = " << reduce_op << "(r , in[reduce_idx]);\n";
                                writer << "reduce_idx += step;\n";
                            }
                            writer.block_end();
                        }
                        for (int64_t j = 0; j < last_r_idx; j++)
                        {
                            writer.block_end();
                        }
                        writer << "out[tid] = r;\n";
                    }
                    writer.block_end();
                }
                writer.block_end();
                return cw;
            }

            LanguageUnit_p codegen_function_definition_acc() { return nullptr; }
            LanguageUnit_p codegen_function_definition_scalar()
            {
                create_ptr(LanguageUnit, cw, codegen_function_name());
                LanguageUnit& writer = *cw;

                uint32_t nthreads = static_cast<uint32_t>(shape_size(input_shape));
                uint32_t n = nthreads;
                uint32_t block_size_x = 1;
                while (n > 1)
                {
                    block_size_x <<= 1;
                    n >>= 1;
                }
                block_size_x = fmin(512, block_size_x);

                writer << "extern \"C\" __global__ void " << writer.symbol << "_scalar("
                       << op->dtypes[0] << "* in, " << op->dtypes[1] << "* out, size_t nthreads)\n";
                writer.block_begin();
                {
                    writer << "extern __shared__ " << op->dtypes[1] << " sdata[];\n";
                    writer << "uint32_t tid = threadIdx.x; \n";
                    writer << "uint32_t step = blockDim.x; \n";
                    writer << "sdata[tid] = 0;\n";
                    writer << "uint32_t in_idx = tid;\n";
                    writer << op->dtypes[1] << " r = 0;\n";
                    writer << "if(in_idx < nthreads)\n";
                    writer.block_begin();
                    writer << "r = in[in_idx];\n";
                    writer << "in_idx += step;\n";
                    writer.block_end();
                    //accumulate reduction to blockDim.x threads
                    uint32_t unroll_num = 8;
                    writer << "while(in_idx + (step * " << unroll_num - 1 << ") < nthreads)\n";
                    writer.block_begin();
                    {
                        for (int i = 0; i < unroll_num; i++)
                        {
                            writer << "r = " << reduce_op << "(r , in[in_idx]);\n";
                            writer << "in_idx += step;\n";
                        }
                    }
                    writer.block_end();
                    writer << "while(in_idx < nthreads)\n";
                    writer.block_begin();
                    {
                        writer << "r = " << reduce_op << "(r , in[in_idx]);\n";
                        writer << "in_idx += step;\n";
                    }
                    writer.block_end();

                    //accumulate 32 threads for each warp
                    for (int i = 16; i >= 1; i >>= 1)
                    {
                        if (block_size_x > i)
                        {
                            writer << "r = " << reduce_op << "(r, __shfl_down_sync(0xffffffff, r, "
                                   << i << ", 32));\n";
                        }
                    }

                    if (block_size_x > 32)
                    {
                        writer << "uint32_t lane_idx = tid & 0x1f; \n";
                        writer << "uint32_t warp_idx = tid >> 5; \n";
                        writer << "if(lane_idx == 0)\n";
                        writer.block_begin();
                        {
                            writer << "sdata[warp_idx] = r;\n";
                        }
                        writer.block_end();
                        writer << "__syncthreads();\n";

                        uint32_t warp_size = block_size_x >> 5;

                        writer << "if(tid < " << warp_size << ")\n";
                        writer.block_begin();
                        {
                            writer << "r = sdata[tid];\n";
                        }
                        writer.block_end();
                        //accumulate 32 threads
                        for (int i = 16; i >= 1; i >>= 1)
                        {
                            if (warp_size > i)
                            {
                                writer << "r = " << reduce_op
                                       << "(r, __shfl_down_sync(0xffffffff, r, " << i
                                       << ", 32));\n";
                            }
                        }
                    }

                    writer << "if(tid == 0)\n";
                    writer.block_begin();
                    {
                        writer << "out[0] = r;\n";
                    }
                    writer.block_end();
                }
                writer.block_end();

                return cw;
            }

            LanguageUnit_p codegen_function_call_nd()
            {
                std::string name = codegen_function_name() + "_nd_call";
                create_ptr(LanguageUnit, cw, name);
                LanguageUnit& writer = *cw;

                uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
                // TODO: currently we set it to 64, will add tuning method later
                uint32_t block_size_x = 64;
                uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);
                writer << codegen_function_name() << "_nd<<<dim3(" << aligned_grid_size_x << ", "
                       << 1 << ", " << 1 << "), dim3(" << block_size_x << ", " << 1 << ", " << 1
                       << "), " << 0 << ", " << 0 << ">>>"
                       << "(" << join(op->arg_names, ", ") << ", " << join(op->out_names, ", ")
                       << ", " << nthreads << ");\n";
                return cw;
            }

            LanguageUnit_p codegen_function_call_scalar()
            {
                std::string name = codegen_function_name() + "_scalar_call";
                create_ptr(LanguageUnit, cw, name);
                LanguageUnit& writer = *cw;

                uint32_t nthreads = static_cast<uint32_t>(shape_size(input_shape));
                uint32_t n = nthreads;
                uint32_t block_size_x = 1;
                while (n > 1)
                {
                    block_size_x <<= 1;
                    n >>= 1;
                }
                block_size_x = fmin(512, block_size_x);
                uint32_t shared_data_bytes = block_size_x * static_cast<uint32_t>(data_bytes);

                writer << codegen_function_name() << "_scalar<<<dim3(" << 1 << ", " << 1 << ", "
                       << 1 << "), dim3(" << block_size_x << ", " << 1 << ", " << 1 << "), "
                       << shared_data_bytes << ", " << 0 << ">>>"
                       << "(" << join(op->arg_names, ", ") << ", " << join(op->out_names, ", ")
                       << ", " << nthreads << ");\n";
                return cw;
            }

            LanguageUnit_p codegen_function_definition() override
            {
                if (out_rank != 0)
                {
                    return codegen_function_definition_nd();
                }
                else
                {
                    uint32_t nthreads = static_cast<uint32_t>(shape_size(input_shape));
                    //if the data size is large, call reduce_to_scalar_acc first and then reduce_to_scalar.
                    //other wise, call reduce to scalar directly.
                    const uint32_t unroll_size = 8;
                    if (nthreads > nthreads_acc * (unroll_size + 1))
                    {
                        //todo(wenxh): Ignore this Case
                        // assert_bool(false) << "no support for GPU memory allocation.";
                    }
                    else
                    {
                        return codegen_function_definition_scalar();
                    }
                }
            }

            LanguageUnit_p codegen_function_call() override
            {
                if (out_rank != 0)
                {
                    return codegen_function_call_nd();
                }
                else
                {
                    uint32_t nthreads = static_cast<uint32_t>(shape_size(input_shape));
                    //if the data size is large, call reduce_to_scalar_acc first and then reduce_to_scalar.
                    //other wise, call reduce to scalar directly.
                    const uint32_t unroll_size = 8;
                    if (nthreads > nthreads_acc * (unroll_size + 1))
                    {
                        assert_bool(false) << "no support for GPU memory allocation.";
                    }
                    else
                    {
                        return codegen_function_call_scalar();
                    }
                }
            }

            LanguageUnit_p codegen_dependency() override
            {
                std::string name = codegen_function_name() + "_dep";
                create_ptr(LanguageUnit, cw, name);

                cw->require(header::cuda);
                cw->require(header::stdio);
                cw->require(macro::MIN);
                cw->require(declaration::num_SMs);

                if (CudaOpMap<T>::math_kernel != nullptr)
                {
                    auto math_kernel = get_math_kernel(
                        reduce_op,
                        CudaOpMap<T>::math_kernel,
                        vector<string>{op->dtypes[0], op->dtypes[0], op->dtypes[1]});
                    assert_nullptr(math_kernel);
                    cw->require(math_kernel);
                }

                return cw;
            }

            static CudaFunction_p codegen(ir::Operator_p inter_op)
            {
                create_ptr(Reduce, cop, inter_op);
                NGRAPH_DEBUG << "Codegen for Reduce function:" << cop->codegen_function_name()
                             << endl;
                return cop;
            }
        };
    }
}
