// Microsoft (c) 2019, Wenxiang
#include "concat.hpp"

cuda::Concat::Concat(ir::Operator_p inter_op)
    : CudaFunction(inter_op)
{
    assert_nullptr(this->op = static_pointer_cast<ir::Concat>(inter_op));
    input_num = op->input_shapes.size();
    split_input_size = 256; //max num of inputs fit 4KB parameter space: 256 * 8 + 7 * ?
    residue = input_num % split_input_size;

    inputs_strides = std::vector<uint32_t>(input_num, 1);
    output_stride = 0;
    concat_axis = op->axis;

    for (size_t i = 0; i < input_num; i++)
    {
        auto arg_rank = op->input_shapes[i].size();
        for (size_t j = concat_axis; j < arg_rank; j++)
        {
            inputs_strides[i] *= op->input_shapes[i][j];
        }
        output_stride += inputs_strides[i];
    }

    block_size_x = 64;
    split_input_stride_offsets.push_back(0);
    split_input_stride_offset = 0;

    for (uint32_t i = 0; i < input_num; i += split_input_size)
    {
        uint32_t nthread = 0;
        uint32_t split_output_stride = 0;
        for (uint32_t j = i; j < i + split_input_size && j < input_num; j++)
        {
            nthread += shape_size(op->input_shapes[j]);
            split_output_stride += inputs_strides[j];
        }
        split_input_stride_offset += split_output_stride;
        split_input_stride_offsets.push_back(split_input_stride_offset);
        split_output_strides.push_back(split_output_stride);
        split_nthreads.push_back(static_cast<uint32_t>(nthread));
        split_aligned_grid_size_x.push_back(
            align_to_block_size(split_nthreads.back(), block_size_x));
    }
}

string cuda::Concat::codegen_function_name()
{
    std::stringstream kernel_name;
    kernel_name << "cuda_concat_" << op->dtype << "_r_" << input_num;

    kernel_name << "_o_" << join(op->output_shape, "_") << "_a_" << concat_axis;
    for (size_t i = 0; i < input_num; i++)
    {
        kernel_name << "_i_" << join(op->input_shapes[i], "_");
    }
    return kernel_name.str();
}

string cuda::Concat::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

LanguageUnit_p cuda::Concat::codegen_function_definition()
{
    LanguageUnit_p cw(new LanguageUnit(codegen_function_name()));
    auto& writer = *cw;

    if (input_num >= split_input_size)
    {
        size_t num_inputs = split_input_size;
        writer << "extern \"C\" __global__ void " << writer.symbol << "_kernel_0(";
        for (size_t i = 0; i < num_inputs; i++)
        {
            writer << op->dtype << "* in" << i << ", ";
        }
        writer << op->dtype << "* out, uint32_t output_stride, uint32_t "
                               "split_output_stride, uint32_t split_input_stride_offset, uint32_t "
                               "input_offset, uint32_t n)\n";
        writer.block_begin();
        {
            writer << "uint32_t inputs_strides[] = {" << join(inputs_strides) << "};\n";
            writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
            writer << "if(tid < n)\n";
            writer.block_begin();
            {
                writer << "uint32_t block_id = tid / split_output_stride;\n";
                writer << "uint32_t block_idx = tid % split_output_stride;\n";
                writer << "uint32_t output_idx = block_id * output_stride + block_idx + "
                          "split_input_stride_offset;\n";
                writer << "out[output_idx] = 1;\n";
                for (size_t i = 0; i < num_inputs; i++)
                {
                    writer << "if(block_idx < inputs_strides[" << i << " + input_offset])\n";
                    writer.block_begin();
                    {
                        writer << "out[output_idx] = in" << i << "[block_id * inputs_strides[" << i
                               << " + input_offset] + block_idx];\n";
                        writer << "return;\n";
                    }
                    writer.block_end();
                    writer << "block_idx -= inputs_strides[" << i << " + input_offset];\n";
                }
            }
            writer.block_end();
        }
        writer.block_end();
    }

    if (residue != 0)
    {
        size_t num_inputs = residue;
        writer << "extern \"C\" __global__ void " << writer.symbol << "_kernel_1(";
        for (size_t i = 0; i < num_inputs; i++)
        {
            writer << op->dtype << "* in" << i << ", ";
        }
        writer << op->dtype << "* out, uint32_t output_stride, uint32_t "
                               "split_output_stride, uint32_t split_input_stride_offset, uint32_t "
                               "input_offset, uint32_t n)\n";
        writer.block_begin();
        {
            writer << "uint32_t inputs_strides[] = {" << join(inputs_strides) << "};\n";
            writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
            writer << "if(tid < n)\n";
            writer.block_begin();
            {
                writer << "uint32_t block_id = tid / split_output_stride;\n";
                writer << "uint32_t block_idx = tid % split_output_stride;\n";
                writer << "uint32_t output_idx = block_id * output_stride + block_idx + "
                          "split_input_stride_offset;\n";
                writer << "out[output_idx] = 1;\n";
                for (size_t i = 0; i < num_inputs; i++)
                {
                    writer << "if(block_idx < inputs_strides[" << i << " + input_offset])\n";
                    writer.block_begin();
                    {
                        writer << "out[output_idx] = in" << i << "[block_id * inputs_strides[" << i
                               << " + input_offset] + block_idx];\n";
                        writer << "return;\n";
                    }
                    writer.block_end();
                    writer << "block_idx -= inputs_strides[" << i << " + input_offset];\n";
                }
            }
            writer.block_end();
        }
        writer.block_end();
    }

    return cw;
}

LanguageUnit_p cuda::Concat::codegen_function_call()
{
    LanguageUnit_p cw(new LanguageUnit(codegen_function_name() + "_call"));
    auto& lu = *cw;

    for (uint32_t i = 0, n = 0; i < input_num; i += split_input_size, n++)
    {
        std::vector<string> args_list;
        for (uint32_t j = i; j < i + split_input_size && j < input_num; j++)
        {
            args_list.push_back(op->arg_names[j]);
        }
        args_list.push_back(op->out_names[0]);
        // args_list.push_back(&param_inputs_strides);
        args_list.push_back(to_string(output_stride));
        args_list.push_back(to_string(split_output_strides[n]));
        args_list.push_back(to_string(split_input_stride_offsets[n]));
        args_list.push_back(to_string(i));
        args_list.push_back(to_string(split_nthreads[n]));
        auto kernel = (args_list.size() == split_input_size + 6) ? "_kernel_0" : "_kernel_1";

        lu << codegen_function_name() << kernel << "<<<dim3(" << split_aligned_grid_size_x[n]
           << ", " << 1 << ", " << 1 << "), dim3(" << block_size_x << ", " << 1 << ", " << 1
           << "), " << 0 << ", " << 0 << ">>>(" << join(args_list) << ");\n";
    }
    return cw;
}

LanguageUnit_p cuda::Concat::codegen_dependency()
{
    LanguageUnit_p cw(new LanguageUnit(codegen_function_name() + "_dep"));
    cw->require(header::cuda);
    return cw;
}

cuda::CudaFunction_p cuda::Concat::codegen(ir::Operator_p inter_op)
{
    Concat_p cop(new Concat(inter_op));
    NGRAPH_DEBUG << "Codegen for Concat function:" << cop->codegen_function_name() << endl;
    return cop;
}