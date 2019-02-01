// Microsoft (c) 2019, Wenxiang
#include "pad.hpp"

cuda::Pad::Pad(ir::Operator_p inter_op)
    : CudaFunction(inter_op)
{
    assert_nullptr(this->op = static_pointer_cast<ir::Pad>(inter_op));
}

string cuda::Pad::codegen_function_name()
{
    std::stringstream kernel_name;
    kernel_name << "pad_" << join(op->dtypes, "_") << op->rank << "pad_i"
                << join(op->input_shape, "_") << "pad_o" << join(op->output_shape) << "_pb"
                << join(op->padding_below, "_") << "_pi" << join(op->padding_interior, "_");
    return kernel_name.str();
}

string cuda::Pad::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

LanguageUnit_p cuda::Pad::codegen_function_definition()
{
    LanguageUnit_p _lu(new LanguageUnit(codegen_function_name()));
    auto& lu = *_lu;
    lu << "extern \"C\" __global__ void cuda_" << lu.symbol << "(" << op->dtypes[0] << "* in, "
       << op->dtypes[1] << "* pad, " << op->dtypes[2] << "* out, "
       << "size_t n)\n";
    lu.block_begin();
    {
        lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        lu << "if (tid < n)\n";
        lu.block_begin();
        {
            auto expand_vector_size = [](string name, vector<size_t>& d) {
                stringstream ss;
                for (int i = 0; i < d.size(); i++)
                    ss << "size_t " << name << i << " = " << to_string(d[i]) << ";\n";
                return ss.str();
            };

            auto expand_vector_uint32 = [](string name, vector<uint32_t>& d) {
                stringstream ss;
                for (int i = 0; i < d.size(); i++)
                    ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
                return ss.str();
            };

            lu << expand_vector_size("input_shape", op->input_shape);
            lu << expand_vector_uint32("input_strides", op->input_strides);
            lu << expand_vector_uint32("output_strides", op->output_strides);
            lu << expand_vector_uint32("pad_below", op->pad_below);
            lu << expand_vector_uint32("pad_interior", op->pad_interior);

            lu << "bool in_bounds = true;\n";
            lu << "uint32_t output_pixel = tid;\n";
            lu << "uint32_t input_pixel = 0;\n";
            lu << "int32_t input, input_dil;\n";

            for (size_t i = 0; i < op->rank; i++)
            {
                if (i != 0)
                {
                    lu << "output_pixel %= output_strides" << i - 1 << ";\n";
                }
                lu << "input_dil = output_pixel / output_strides" << i << " - padding_below" << i
                   << ";\n";

                lu << "input = input_dil / (padding_interior" << i << " + 1);\n";
                lu << "input_dil %= (padding_interior" << i << " + 1);\n";
                lu << "in_bounds = in_bounds && (input >= 0) && (input < input_shape" << i
                   << ") && (input_dil == 0);\n";
                lu << "input_pixel += input * input_strides" << i << ";\n";
            }
            lu << "out[tid] = (in_bounds) ? in[input_pixel] : *pad;\n";
        }
        lu.block_end();
    }
    lu.block_end();
    return _lu;
}

LanguageUnit_p cuda::Pad::codegen_function_call()
{
    LanguageUnit_p _lu(new LanguageUnit(codegen_function_name() + "_call"));
    auto& lu = *_lu;
    uint32_t nthreads = static_cast<uint32_t>(shape_size(op->output_shape));
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);
    lu << codegen_function_name() << "<<<dim3(" << aligned_grid_size_x << ", " << 1 << ", " << 1
       << "), dim3(" << block_size_x << ", " << 1 << ", " << 1 << "), " << 0 << ", " << 0 << ">>>"
       << "(" << join(op->arg_names, ", ") << ", " << join(op->out_names, ", ") << ", " << nthreads
       << ");\n";
    return _lu;
}

LanguageUnit_p cuda::Pad::codegen_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(codegen_function_name() + "_dep"));
    return _lu;
}

cuda::CudaFunction_p cuda::Pad::codegen(ir::Operator_p inter_op)
{
    Pad_p cop(new Pad(inter_op));
    NGRAPH_DEBUG << "Codegen for Pad function:" << cop->codegen_function_name() << endl;
    return cop;
}