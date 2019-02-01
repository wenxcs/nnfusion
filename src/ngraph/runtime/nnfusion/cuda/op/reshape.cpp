// Microsoft (c) 2019, Wenxiang
#include "reshape.hpp"
#include "noop.hpp"
#include "result.hpp"

/*Common methods for Reshape*D classes*/
cuda::Reshape::Reshape(ir::Operator_p inter_op)
    : CudaFunction(inter_op)
{
    assert_nullptr(this->op = static_pointer_cast<ir::Reshape>(inter_op));
}

cuda::CudaFunction_p cuda::Reshape::codegen(ir::Operator_p inter_op)
{
    auto irop = static_pointer_cast<ir::Reshape>(inter_op);
    assert_nullptr(irop) << "Input operator is invalid.";
    NGRAPH_DEBUG << "Codegen for Reshape function:" << irop->arg_rank << endl;

    // <TODO> make noop & memcpy in other place
    if (irop->isNoop())
    {
        create_ptr(Noop, cop, inter_op);
        NGRAPH_DEBUG << cop->codegen_function_name() << endl;
        return cop;
    }
    if (irop->isMemcpy())
    {
        create_ptr(Result, cop, inter_op);
        NGRAPH_DEBUG << cop->codegen_function_name() << endl;
        return cop;
    }
    assert_bool(!irop->isNoop() && !irop->isMemcpy()) << "Wrong Codegen for this operator.";
    if (irop->arg_rank == 2)
    {
        create_ptr(Reshape2D, cop, inter_op);
        NGRAPH_DEBUG << cop->codegen_function_name() << endl;
        return cop;
    }
    else if (irop->arg_rank == 3)
    {
        create_ptr(Reshape3D, cop, inter_op);
        NGRAPH_DEBUG << cop->codegen_function_name() << endl;
        return cop;
    }
    else
    {
        create_ptr(ReshapehD, cop, inter_op);
        NGRAPH_DEBUG << cop->codegen_function_name() << endl;
        return cop;
    }
}

string cuda::Reshape::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

LanguageUnit_p cuda::Reshape::codegen_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(codegen_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

/*Reshape Subclasses*/
cuda::Reshape2D::Reshape2D(ir::Operator_p inter_op)
    : Reshape(inter_op)
{
    // <TODO> currently we set it to 16, will add tuning method later
    block_size = 16;
    input_strides = row_major_strides(op->arg_shape);
    output_strides = ngraph::NVShape(op->arg_rank);
    trans_strides = ngraph::NVShape(op->arg_rank);
    int stride = 1;
    for (int64_t i = op->arg_rank - 1; i >= 0; i--)
    {
        output_strides[i] = stride;
        stride *= op->arg_shape[op->input_order[i]];
    }
    for (int64_t i = 0; i < op->arg_rank; i++)
    {
        trans_strides[op->input_order[i]] = output_strides[i];
    }
}

string cuda::Reshape2D::codegen_function_name()
{
    std::stringstream kernel_name;
    kernel_name << "cuda_reshape_" << join(op->dtypes, "_") << "_i_" << join(op->arg_shape, "_")
                << "_o_" << join(op->input_order, "_");
    return kernel_name.str();
}

LanguageUnit_p cuda::Reshape2D::codegen_function_definition()
{
    create_ptr(LanguageUnit, plu, codegen_function_name());
    auto& lu = *plu;
    auto& data_type = op->dtypes[1];
    lu << "extern \"C\" __global__ void " << lu.symbol << "(";
    lu << op->dtypes[0] << "* in, ";
    lu << op->dtypes[1] << "* out)\n";
    lu.block_begin();
    {
        // Common data area starts
        auto expand_vector_uint32 = [](string name, vector<uint32_t>& d) {
            stringstream ss;
            for (int i = 0; i < d.size(); i++)
                ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
            return ss.str();
        };

        lu << expand_vector_uint32("input_strides", input_strides);
        lu << expand_vector_uint32("trans_strides", trans_strides);
        lu << "size_t nx = " << op->arg_shape[1] << ";\n";
        lu << "size_t ny = " << op->arg_shape[0] << ";\n";
        // Common data area ends

        lu << "__shared__ " << data_type << " tile[" << block_size << "][" << block_size + 1
           << "];\n";
        lu << "uint32_t base1 = blockIdx.x * blockDim.x;\n";
        lu << "uint32_t base0 = blockIdx.y * blockDim.y;\n";
        lu << "uint32_t tid1 = threadIdx.x;\n";
        lu << "uint32_t tid0 = threadIdx.y;\n";
        lu << "uint32_t idx1 = base1 + tid1;\n";
        lu << "uint32_t idx0 = base0 + tid0;\n";

        lu << "if (idx1 < nx && idx0 < ny)\n";
        lu.block_begin();
        {
            lu << "uint32_t input_idx = 0;\n";
            for (int i = 0; i < 2; i++)
            {
                lu << "input_idx += input_strides" << i << "* idx" << i << ";\n";
            }
            lu << "tile[tid0][tid1] = in[input_idx];\n";
        }
        lu.block_end();

        lu << "idx1 = base1 + tid0;\n";
        lu << "idx0 = base0 + tid1;\n";
        lu << "__syncthreads();\n";

        lu << "if (idx1 < nx && idx0 < ny)\n";
        lu.block_begin();
        {
            lu << "uint32_t output_idx = 0;\n";
            for (int i = 0; i < 2; i++)
            {
                lu << "output_idx += trans_strides" << i << "* idx" << i << ";\n";
            }
            lu << "out[output_idx] = tile[tid1][tid0];\n";
        }
        lu.block_end();
    }
    lu.block_end();
    return plu;
}

LanguageUnit_p cuda::Reshape2D::codegen_function_call()
{
    create_ptr(LanguageUnit, plu, codegen_function_name() + "_call");
    auto& lu = *plu;
    uint32_t aligned_grid_size_x = align_to_block_size(op->arg_shape[1], block_size);
    uint32_t aligned_grid_size_y = align_to_block_size(op->arg_shape[0], block_size);

    lu << codegen_function_name() << "<<<dim3(" << aligned_grid_size_x << ", "
       << aligned_grid_size_y << ", " << 1 << "), dim3(" << block_size << ", " << block_size << ", "
       << 1 << "), " << 0 << ", " << 0 << ">>>"
       << "(" << join(op->arg_names, ", ") << ", " << join(op->out_names, ", ") << ");\n";

    return plu;
}

cuda::Reshape3D::Reshape3D(ir::Operator_p inter_op)
    : Reshape(inter_op)
{
    block_size = std::vector<uint32_t>(3, 0);
    // TODO: currently we set it to 16, will add tuning method later
    block_size_x = 16;
    block_size[0] = block_size_x;                                       //x
    block_size[2] = (op->input_order[2] == 0) ? block_size_x : 1;       //z
    block_size[1] = (block_size[2] == block_size_x) ? 1 : block_size_x; //y
    input_strides = ngraph::row_major_strides(op->arg_shape);
    output_strides = ngraph::NVShape(op->arg_rank);
    trans_strides = ngraph::NVShape(op->arg_rank);
    int stride = 1;
    for (int64_t i = op->arg_rank - 1; i >= 0; i--)
    {
        output_strides[i] = stride;
        stride *= op->arg_shape[op->input_order[i]];
    }
    for (int64_t i = 0; i < op->arg_rank; i++)
    {
        trans_strides[op->input_order[i]] = output_strides[i];
    }
}

string cuda::Reshape3D::codegen_function_name()
{
    std::stringstream kernel_name;
    kernel_name << "cuda_reshape_" << join(op->dtypes, "_") << "_r_" << join(op->input_order, "_")
                << "_i_" + join(op->arg_shape, "_");
    return kernel_name.str();
}

LanguageUnit_p cuda::Reshape3D::codegen_function_definition()
{
    create_ptr(LanguageUnit, plu, codegen_function_name());
    auto& lu = *plu;
    auto& data_type = op->dtypes[1];

    lu << "extern \"C\" __global__ void " << lu.symbol << "(";
    lu << op->dtypes[0] << "* in, ";
    lu << op->dtypes[1] << "* out)\n";
    lu.block_begin();
    {
        // Common data area starts
        auto expand_vector_uint32 = [](string name, vector<uint32_t>& d) {
            stringstream ss;
            for (int i = 0; i < d.size(); i++)
                ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
            return ss.str();
        };

        lu << expand_vector_uint32("input_strides", input_strides);
        lu << expand_vector_uint32("trans_strides", trans_strides);
        lu << "size_t nx = " << op->arg_shape[2] << ";\n";
        lu << "size_t ny = " << op->arg_shape[1] << ";\n";
        lu << "size_t nz = " << op->arg_shape[0] << ";\n";
        // Common data area ends

        lu << "__shared__ " << data_type << " tile[" << block_size[2] << "][" << block_size[1]
           << "][" << block_size[0] + 1 << "];\n";
        lu << "uint32_t base2 = blockIdx.x * blockDim.x;\n";
        lu << "uint32_t base1 = blockIdx.y * blockDim.y;\n";
        lu << "uint32_t base0 = blockIdx.z * blockDim.z;\n";
        lu << "uint32_t tid2 = threadIdx.x;\n";
        lu << "uint32_t tid1 = threadIdx.y;\n";
        lu << "uint32_t tid0 = threadIdx.z;\n";
        lu << "uint32_t otid2 = tid2;\n";
        lu << "uint32_t otid1 = tid1;\n";
        lu << "uint32_t otid0 = tid0;\n";
        lu << "uint32_t idx2 = base2 + tid2;\n";
        lu << "uint32_t idx1 = base1 + tid1;\n";
        lu << "uint32_t idx0 = base0 + tid0;\n";

        lu << "if (idx2 < nx && idx1 < ny && idx0 < nz)\n";
        lu.block_begin();
        {
            lu << "uint32_t input_idx = 0;\n";
            for (int i = 0; i < 3; i++)
            {
                lu << "input_idx += input_strides" << i << "* idx" << i << ";\n";
            }
            lu << "tile[tid0][tid1][tid2] = in[input_idx];\n";
        }
        lu.block_end();

        if (op->input_order[2] == 1)
        {
            lu << "otid2 = tid1;\n";
            lu << "otid1 = tid2;\n";
        }
        else if (op->input_order[2] == 0)
        {
            lu << "otid2 = tid0;\n";
            lu << "otid0 = tid2;\n";
        }
        lu << "idx2 = base2 + otid2;\n";
        lu << "idx1 = base1 + otid1;\n";
        lu << "idx0 = base0 + otid0;\n";
        lu << "__syncthreads();\n";

        lu << "if (idx2 < nx && idx1 < ny && idx0 < nz)\n";
        lu.block_begin();
        {
            lu << "uint32_t output_idx = 0;\n";
            for (int i = 0; i < 3; i++)
            {
                lu << "output_idx += trans_strides" << i << "* idx" << i << ";\n";
            }
            lu << "out[output_idx] = tile[otid0][otid1][otid2];\n";
        }
        lu.block_end();
    }
    lu.block_end();

    return plu;
}

LanguageUnit_p cuda::Reshape3D::codegen_function_call()
{
    uint32_t aligned_grid_size_x = align_to_block_size(op->arg_shape[2], block_size[0]);
    uint32_t aligned_grid_size_y = align_to_block_size(op->arg_shape[1], block_size[1]);
    uint32_t aligned_grid_size_z = align_to_block_size(op->arg_shape[0], block_size[2]);

    create_ptr(LanguageUnit, plu, codegen_function_name() + "_call");
    auto& lu = *plu;

    lu << codegen_function_name() << "<<<dim3(" << aligned_grid_size_x << ", "
       << aligned_grid_size_y << ", " << aligned_grid_size_z << "), dim3(" << block_size[0] << ", "
       << block_size[1] << ", " << block_size[2] << "), " << 0 << ", " << 0 << ">>>"
       << "(" << join(op->arg_names, ", ") << ", " << join(op->out_names, ", ") << ");\n";

    return plu;
}

cuda::ReshapehD::ReshapehD(ir::Operator_p inter_op)
    : Reshape(inter_op)
{
    block_size_x = 64;
    input_strides = ngraph::row_major_strides(op->arg_shape);
    output_strides = ngraph::NVShape(op->arg_rank);
    trans_strides = ngraph::NVShape(op->arg_rank);
    int stride = 1;
    for (int64_t i = op->arg_rank - 1; i >= 0; i--)
    {
        output_strides[i] = stride;
        stride *= op->arg_shape[op->input_order[i]];
    }
    for (int64_t i = 0; i < op->arg_rank; i++)
    {
        trans_strides[op->input_order[i]] = output_strides[i];
    }
}

string cuda::ReshapehD::codegen_function_name()
{
    std::stringstream kernel_name;
    kernel_name << "cuda_reshape_" << join(op->dtypes, "_") << "_r_" << op->arg_rank << "_i_"
                << join(op->arg_shape, "_") << "_o_" << join(op->input_order, "_");
    return kernel_name.str();
}

LanguageUnit_p cuda::ReshapehD::codegen_function_definition()
{
    create_ptr(LanguageUnit, plu, codegen_function_name());
    auto& lu = *plu;
    uint32_t nthreads = static_cast<uint32_t>(shape_size(op->arg_shape));

    lu << "extern \"C\" __global__ void " << lu.symbol << "(";
    lu << op->dtypes[0] << "* in, ";
    lu << op->dtypes[1] << "* out)\n";
    lu.block_begin();
    {
        // Common data area starts
        auto expand_vector_uint32 = [](string name, vector<uint32_t>& d) {
            stringstream ss;
            for (int i = 0; i < d.size(); i++)
                ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
            return ss.str();
        };

        lu << expand_vector_uint32("input_strides", input_strides);
        lu << expand_vector_uint32("trans_strides", trans_strides);
        lu << "size_t n = " << nthreads << ";\n";
        // Common data area ends

        lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        lu << "if (tid < n)\n";
        lu.block_begin();
        {
            lu << "uint32_t input_idx = tid;\n";
            lu << "uint32_t output_idx = 0;\n";
            size_t i = 0;
            for (; i < op->arg_rank - 1; i++)
            {
                lu << "output_idx += (input_idx / input_strides" << i << ") * trans_strides" << i
                   << ";\n";
                lu << "input_idx %= input_strides" << i << ";\n";
            }
            lu << "output_idx += (input_idx / input_strides" << i << ") * trans_strides" << i
               << ";\n";
            lu << "out[output_idx] = in[tid];\n";
        }
        lu.block_end();
    }
    lu.block_end();
    return plu;
}

LanguageUnit_p cuda::ReshapehD::codegen_function_call()
{
    uint32_t nthreads = static_cast<uint32_t>(shape_size(op->arg_shape));
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    create_ptr(LanguageUnit, plu, codegen_function_name() + "_call");
    auto& lu = *plu;

    lu << codegen_function_name() << "<<<dim3(" << aligned_grid_size_x << ", " << 1 << ", " << 1
       << "), dim3(" << block_size_x << ", " << 1 << ", " << 1 << "), " << 0 << ", " << 0 << ">>>"
       << "(" << join(op->arg_names, ", ") << ", " << join(op->out_names, ", ") << ");\n";

    return plu;
}