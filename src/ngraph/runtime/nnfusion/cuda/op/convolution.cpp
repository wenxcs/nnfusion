// Microsoft (c) 2019, Wenxiang
#include "convolution.hpp"
#include "../cuda_cudnn.hpp"

cuda::Convolution::Convolution(ir::Operator_p inter_op)
    : CudaFunction(inter_op)
{
    enforce_not_nullptr(this->op = static_pointer_cast<ir::Convolution>(inter_op));
}

LanguageUnit_p cuda::Convolution::codegen_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(codegen_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

cuda::CudaFunction_p cuda::Convolution::codegen(ir::Operator_p inter_op)
{
    auto convolution = static_pointer_cast<ir::Convolution>(inter_op);
    enforce_not_nullptr(convolution);

    if (convolution->padding_below_diff.size() > 3)
    {
        // <TODO> Support Conv3D or others.
        enforce(false) << "Current we only support Conv2D";
        create_ptr(ConvolutionCuda, cop, inter_op);
        LOG_INFO << "Codegen for Cuda Convolution." << endl;
        return cop;
    }
    else
    {
        create_ptr(ConvolutionCudnn, cop, inter_op);
        LOG_INFO << "Codegen for Cudnn Convolution." << endl;
        return cop;
    }
}

string cuda::Convolution::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

string cuda::ConvolutionCudnn::codegen_function_name()
{
    std::stringstream ss;
    ss << "cudnn_convolution_op_" << op->dtype << "_i" << join(op->input_shape, "_") << "_w"
       << join(op->filter_shape, "_") << "_o" << join(op->output_shape, "_") << "_ws"
       << join(op->window_movement_strides, "_") << "_wd" << join(op->window_dilation_strides, "_")
       << "_p" << join(op->padding_below_diff, "_");
    return ss.str();
}

cuda::ConvolutionCudnn::ConvolutionCudnn(ir::Operator_p inter_op)
    : Convolution(inter_op)
{
    // <TODO> full feature of cudnn convolutoin
    bool is_deconvolution = false;
    for (auto a : op->data_dilation_strides)
    {
        if (a != 1)
        {
            is_deconvolution = true;
            break;
        }
    }
    enforce(is_deconvolution == false) << "Deconvolution is not supported by now.";
    bool pad_required = (op->padding_below_diff != op->padding_above_diff);
    enforce(pad_required == false) << "Asymetric padding is not supported by now.";
}

LanguageUnit_p cuda::ConvolutionCudnn::codegen_function_definition()
{
    create_ptr(LanguageUnit, plu, codegen_function_name());
    auto& lu = *plu;
    lu.require(macro::CUDNN_SAFE_CALL);
    lu.require(macro::CUDA_SAFE_CALL);

    Shape padding_below(op->padding_below_diff.size(), 0);

    for (int i = 0; i < padding_below.size(); i++)
    {
        padding_below[i] = static_cast<size_t>(op->padding_below_diff[i]);
    }

    lu << "void " << lu.symbol << "(cudnnHandle_t cudnn_handle, " << op->dtypes[0] << "* in, "
       << op->dtypes[1] << "* filter, " << op->dtypes[2] << "* out)\n";
    lu.block_begin();
    {
        lu << "cudnnDataType_t data_type = " << get_cudnn_datatype(op->dtype) << ";\n";
        lu << cudnn_tensor_descriptor_from_shape(op->input_shape, "tensor_desc_0")->get_code();
        lu << cudnn_tensor_descriptor_from_shape(op->output_shape, "tensor_desc_1")->get_code();
        lu << get_cudnn_filter_descriptor(op->filter_shape, "filter_desc")->get_code();
        lu << get_cudnn_convolution_descriptor(padding_below,
                                               op->window_movement_strides,
                                               op->window_dilation_strides,
                                               "conv_desc")
                  ->get_code();
        // <TODO> Support different cudnnConvolutionFwdAlgo_t
        lu << "// <TODO> Support different cudnnConvolutionFwdAlgo_t.\n";
        lu << "cudnnConvolutionFwdAlgo_t conv_fwd_algo = "
              "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;\n";
        lu << "const float alpha = 1.0;\n";
        lu << "const float beta = 0.0;\n";
        lu << "size_t workspace_size_in_bytes = 0;\n";
        lu << "CUDNN_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, "
           << "tensor_desc_0, "
           << "filter_desc, "
           << "conv_desc, "
           << "tensor_desc_1, "
           << "conv_fwd_algo, "
           << "&workspace_size_in_bytes));\n";
        lu << "void *workspace_ptr;\n"
           << "CUDA_SAFE_CALL(cudaMalloc(&workspace_ptr, workspace_size_in_bytes));\n";
        lu << "CUDNN_SAFE_CALL(cudnnConvolutionForward(cudnn_handle, "
           << "&alpha, "
           << "tensor_desc_0, "
           << "in,"
           << "filter_desc, "
           << "filter, "
           << "conv_desc, "
           << "conv_fwd_algo, "
           << "workspace_ptr, "
           << "workspace_size_in_bytes, "
           << "&beta, "
           << "tensor_desc_1, "
           << "out));\n";
        lu << "CUDA_SAFE_CALL(cudaFree(workspace_ptr));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));\n";
    }
    lu.block_end();
    return plu;
}

LanguageUnit_p cuda::ConvolutionCudnn::codegen_function_call()
{
    create_ptr(LanguageUnit, plu, codegen_function_name());
    auto& lu = *plu;
    lu << codegen_function_name() << "(global_cudnn_handle, " << op->arg_names[0] << ", "
       << op->arg_names[1] << ", " << op->out_names[0] << ");\n";
    return plu;
}

LanguageUnit_p cuda::ConvolutionCudnn::codegen_dependency()
{
    create_ptr(LanguageUnit, plu, codegen_function_name());
    plu->require(header::cudnn);
    plu->require(header::cuda);
    plu->require(declaration::global_cudnn_handle);
    return plu;
}

cuda::ConvolutionCuda::ConvolutionCuda(ir::Operator_p inter_op)
    : Convolution(inter_op)
{
}

string cuda::ConvolutionCuda::codegen_function_name()
{
    return "conv2d";
}

LanguageUnit_p cuda::ConvolutionCuda::codegen_function_definition()
{
    create_ptr(LanguageUnit, plu, codegen_function_name());
    return plu;
}

LanguageUnit_p cuda::ConvolutionCuda::codegen_function_call()
{
    create_ptr(LanguageUnit, plu, codegen_function_name() + "_call");
    return plu;
}

LanguageUnit_p cuda::ConvolutionCuda::codegen_dependency()
{
    create_ptr(LanguageUnit, plu, codegen_function_name() + "_dep");
    return plu;
}
