// Microsoft (c) 2019, Wenxiang
#include "slice.hpp"

cuda::Slice::Slice(ir::Operator_p inter_op)
    : CudaFunction(inter_op)
{
    assert_nullptr(this->op = static_pointer_cast<ir::Slice>(inter_op));

    nthreads = static_cast<uint32_t>(shape_size(op->output_shape));
    // TODO: currently we set it to 64, will add tuning method later
    block_size_x = 64;
    aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);
    output_strides = row_major_strides(op->output_shape);
    input_strides = row_major_strides(op->input_shape);
}

string cuda::Slice::codegen_function_name()
{
    std::stringstream kernel_name;
    kernel_name << "cuda_slice_" << join(op->dtypes, "_") << "_r_" << op->output_shape.size()
                << "_i_" << join(op->input_shape, "_") << "_o_" << join(op->output_shape, "_")
                << "_lb_" << join(op->lower_bounds, "_") << "_ss_" << join(op->slice_strides, "_");
    return kernel_name.str();
}

string cuda::Slice::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

LanguageUnit_p cuda::Slice::codegen_function_definition()
{
    LanguageUnit_p cw(new LanguageUnit(codegen_function_name()));
    auto& writer = *cw;
    writer << "extern \"C\" __global__ void " << writer.symbol << "(" << op->dtypes[0] << "* in, "
           << op->dtypes[1] << "* out, uint32_t n)\n";
    writer.block_begin();
    {
        writer << "uint32_t input_strides[] = {" << join(input_strides) << "};\n";
        writer << "uint32_t output_strides[] = {" << join(output_strides) << "};\n";
        writer << "uint32_t lower_bounds[] = {" << join(op->lower_bounds) << "};\n";
        writer << "uint32_t slice_strides[] = {" << join(op->slice_strides) << "};\n";

        writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < n)\n";
        writer.block_begin();
        {
            writer << "uint32_t input_idx = 0;\n";
            writer << "uint32_t output_idx = tid;\n";
            size_t i = 0;
            for (; i < op->output_shape.size() - 1; i++)
            {
                writer << "input_idx += (((output_idx / output_strides[" << i
                       << "]) * slice_strides[" << i << "]) + "
                                                        "lower_bounds["
                       << i << "]) * input_strides[" << i << "];\n";
                writer << "output_idx %= output_strides[" << i << "];\n";
            }
            writer << "input_idx += (((output_idx / output_strides[" << i << "]) * slice_strides["
                   << i << "]) + "
                           "lower_bounds["
                   << i << "]) * input_strides[" << i << "];\n";
            writer << "out[tid] = in[input_idx];\n";
        }

        writer.block_end();
    }
    writer.block_end();
    return cw;
}

LanguageUnit_p cuda::Slice::codegen_function_call()
{
    LanguageUnit_p cw(new LanguageUnit(codegen_function_name() + "_call"));
    auto& lu = *cw;
    lu << codegen_function_name() << "<<<dim3(" << aligned_grid_size_x << ", " << 1 << ", " << 1
       << "), dim3(" << block_size_x << ", " << 1 << ", " << 1 << "), " << 0 << ", " << 0 << ">>>"
       << "(" << join(op->arg_names, ", ") << ", " << join(op->out_names, ", ") << ", " << nthreads
       << ");\n";
    return cw;
}

LanguageUnit_p cuda::Slice::codegen_dependency()
{
    LanguageUnit_p cw(new LanguageUnit(codegen_function_name() + "_dep"));
    cw->require(header::cuda);
    return cw;
}

cuda::CudaFunction_p cuda::Slice::codegen(ir::Operator_p inter_op)
{
    Slice_p cop(new Slice(inter_op));
    NGRAPH_DEBUG << "Codegen for Slice function:" << cop->codegen_function_name() << endl;
    return cop;
}