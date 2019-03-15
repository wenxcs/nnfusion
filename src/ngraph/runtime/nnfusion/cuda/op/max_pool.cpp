// Microsoft (c) 2019, Wenxiang
#include "max_pool.hpp"
#include "../../core/type_info.hpp"
#include "../cuda_cudnn.hpp"

cuda::MaxPool::MaxPool(ir::Operator_p inter_op)
    : CudaFunction(inter_op)
{
    assert_nullptr(this->op = static_pointer_cast<ir::MaxPool>(inter_op));
}

cuda::CudaFunction_p cuda::MaxPool::codegen(ir::Operator_p inter_op)
{
    auto op = static_pointer_cast<ir::MaxPool>(inter_op);
    assert_nullptr(op);
    if (op->input_shape.size() == 3)
    {
        // MaxPool1d of cuda code
        MaxPool_p cop(new MaxPool1D(inter_op));
        NGRAPH_DEBUG << "Codegen for MaxPool function:" << cop->codegen_function_name() << endl;
        return cop;
    }
    else
    {
        // MaxPoolmD for cudnn code
        create_ptr(MaxPoolmD, max_pool_op, inter_op);
        return max_pool_op;
    }
}

cuda::MaxPool1D::MaxPool1D(ir::Operator_p inter_op)
    : MaxPool(inter_op)
{
    assert_bool(op->padding_below == op->padding_above)
        << "currently don't suport asymetric padding!";

    window_width = op->window_shape.back();
    window_stride = op->window_stride.back();
    input_width = op->input_shape.back();
    output_width = op->output_shape.back();
}

string cuda::MaxPool1D::codegen_function_name()
{
    std::string kernel_name = "cuda_maxpool_" + join(op->dtypes, "_") + "_iw" +
                              std::to_string(input_width) + "_ow" + std::to_string(output_width) +
                              "_ww" + std::to_string(window_width) + "_wst" +
                              std::to_string(window_stride);
    std::replace(kernel_name.begin(), kernel_name.end(), ' ', '_');

    return kernel_name;
}

string cuda::MaxPool1D::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

LanguageUnit_p cuda::MaxPool1D::codegen_function_definition()
{
    // params
    const std::string name = codegen_function_name();
    create_ptr(LanguageUnit, _lu, codegen_function_name());
    auto& lu = *_lu;

    // assumes data is in NCW format
    lu << "extern \"C\" __global__ void " << name << "(" << op->dtypes.front() << "* in, "
       << op->dtypes.back() << "* out, size_t nthreads)\n";
    lu.block_begin();
    {
        // index into output tensor
        lu << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        lu << "if (tid < nthreads)\n";
        lu.block_begin();
        {
            // index into input tensor
            lu << "size_t start = (tid / " << output_width << ") * " << input_width << " + "
               << " (tid % " << output_width << ") * " << window_stride << ";\n";
            lu << op->dtypes[0] << " max_val = " << TypeInfo::Get(op->dtypes[0])->lowest() << ";\n";
            lu << "for (size_t i = start; i < start + " << window_width << "; i++)\n";
            lu.block_begin();
            {
                lu << "const " << op->dtypes[0] << " input = in[i];\n";
                lu << "if (input > max_val)\n";
                lu.block_begin();
                {
                    lu << "max_val = input;\n";
                }
                lu.block_end();
            }
            lu.block_end();
            lu << "out[tid] = max_val;\n";
        }
        lu.block_end();
    }
    lu.block_end();
    return _lu;
}

LanguageUnit_p cuda::MaxPool1D::codegen_function_call()
{
    LanguageUnit_p lu(new LanguageUnit(codegen_function_name() + "_call"));
    size_t nthreads = shape_size(op->output_shape);
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x =
        align_to_block_size(static_cast<uint32_t>(nthreads), block_size_x);

    *lu << codegen_function_name() << "<<<dim3(" << aligned_grid_size_x << ", " << 1 << ", " << 1
        << "), dim3(" << block_size_x << ", " << 1 << ", " << 1 << "), " << 0 << ", " << 0 << ">>>"
        << "(" << join(op->arg_names, ", ") << ", " << join(op->out_names, ", ") << ", " << nthreads
        << ");\n";

    return lu;
}

LanguageUnit_p cuda::MaxPool1D::codegen_dependency()
{
    LanguageUnit_p cw(new LanguageUnit);
    cw->require(header::cuda);
    return cw;
}

string cuda::MaxPoolmD::codegen_function_name()
{
    std::stringstream ss;
    string dtype = op->out[0].get_element_type().c_type_string();
    ss << "cudnn_maxpool_dtype_" << dtype << "_i" << join(op->input_shape, "_") << "_o"
       << join(op->output_shape, "_") << "_ws" << join(op->window_shape, "_") << "_wst"
       << join(op->window_stride, "_") << "_pb" << join(op->padding_below, "_") << "_pb"
       << join(op->padding_above, "_");
    return ss.str();
}

string cuda::MaxPoolmD::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

LanguageUnit_p cuda::MaxPoolmD::codegen_function_definition()
{
    assert_bool(op->input_shape.size() == 4 || op->input_shape.size() == 5)
        << "Cudnn Pooling wrong input.";
    create_ptr(LanguageUnit, plu, codegen_function_name());
    auto& lu = *plu;
    lu.require(macro::CUDNN_SAFE_CALL);

    lu << "void " << lu.symbol << "(cudnnHandle_t cudnn_handle, " << op->dtypes[0] << "* in, "
       << op->dtypes[1] << "* out)\n";
    lu.block_begin();
    {
        auto input_desc = cudnn_tensor_descriptor_from_shape(op->input_shape, "input_desc");
        auto output_desc = cudnn_tensor_descriptor_from_shape(op->output_shape, "output_desc");

        lu << input_desc->get_code();
        lu << output_desc->get_code();

        lu << "cudnnPoolingDescriptor_t desc;\n";
        lu << "cudnnCreatePoolingDescriptor(&desc);\n";
        if (op->input_shape.size() == 4)
        {
            lu << "CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc,"
               << " CUDNN_POOLING_MAX,"
               << " CUDNN_NOT_PROPAGATE_NAN," << static_cast<int>(op->window_shape[0]) << ", "
               << static_cast<int>(op->window_shape[1]) << ", "
               << static_cast<int>(op->padding_below[0]) << ", "
               << static_cast<int>(op->padding_below[1]) << ", "
               << static_cast<int>(op->window_stride[0]) << ", "
               << static_cast<int>(op->window_stride[1]) << "));\n";
        }
        else /*op->input_shape.size() == 5*/
        {
            std::vector<int> w_strides(op->window_stride.size());
            std::vector<int> w_shape(op->window_shape.size());
            std::vector<int> w_padding(op->padding_below.size());
            for (int i = 0; i < op->window_shape.size(); i++)
            {
                w_shape[i] = static_cast<int>(op->window_shape[i]);
                w_strides[i] = static_cast<int>(op->window_stride[i]);
                w_padding[i] = static_cast<int>(op->padding_below[i]);
            }

            auto expand_vector_int = [](string name, vector<int>& d) {
                stringstream ss;
                assert_bool(d.size() > 0);
                ss << "int " << name << "[] = {";
                for (int i = 0; i + 1 < d.size(); i++)
                    ss << to_string(d[i]) << ", ";
                ss << to_string(d.back()) << "}\n";
                return ss.str();
            };

            lu << expand_vector_int("w_shape", w_shape);
            lu << expand_vector_int("w_strides", w_strides);
            lu << expand_vector_int("w_padding", w_padding);

            lu << "CUDNN_SAFE_CALL(cudnnSetPoolingNdDescriptor(desc, "
               << "CUDNN_POOLING_MAX, "
               << "CUDNN_NOT_PROPAGATE_NAN, "
               << "3, w_shape, w_padding, w_strides));\n";
        }

        lu << "const float alpha = 1.0;\n";
        lu << "const float beta = 0.0;\n";

        lu << "CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle,"
           << " desc,"
           << " &alpha,"
           << " input_desc,"
           << " in,"
           << " &beta,"
           << " output_desc,"
           << " out));\n";

        lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));\n";
    }
    lu.block_end();
    return plu;
}

LanguageUnit_p cuda::MaxPoolmD::codegen_function_call()
{
    create_ptr(LanguageUnit, plu, codegen_function_name());
    auto& lu = *plu;
    lu << codegen_function_name() << "(global_cudnn_handle, " << op->arg_names[0] << ", "
       << op->out_names[0] << ");\n";
    return plu;
}

LanguageUnit_p cuda::MaxPoolmD::codegen_dependency()
{
    create_ptr(LanguageUnit, _lu, codegen_function_name() + "_dep");
    auto& lu = *_lu;
    lu.require(header::cudnn);
    lu.require(declaration::global_cudnn_handle);
    return _lu;
}