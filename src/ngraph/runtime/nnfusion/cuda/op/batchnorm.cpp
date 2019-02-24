// Microsoft (c) 2019, Wenxiang
#include "batchnorm.hpp"
#include "../cuda_cudnn.hpp"

cuda::BatchNorm::BatchNorm(ir::Operator_p inter_op)
    : CudaFunction(inter_op)
{
    assert_nullptr(this->op = static_pointer_cast<ir::BatchNorm>(inter_op));
    if (op->epsilon < 1e-5)
    {
        throw std::runtime_error("Batch Norm epsilon is less than 1e-5");
    }
}

string cuda::BatchNorm::codegen_function_name()
{
    // Assumes NC{d1...dN} format
    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::digits10 + 2);

    ss << "cudnn_bn_op"
       << "_dtype_" << op->dtype << "_ts" << join(op->tensor_shape, "_") << "_ps"
       << join(op->param_shape, "_");
    std::string hash = ss.str();
    std::replace(hash.begin(), hash.end(), '.', '_');
    return hash;
}

string cuda::BatchNorm::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

LanguageUnit_p cuda::BatchNorm::codegen_function_definition()
{
    create_ptr(LanguageUnit, plu, codegen_function_name());
    auto& lu = *plu;
    lu.require(macro::CUDNN_SAFE_CALL);

    lu << "void " << lu.symbol << "(cudnnHandle_t cudnn_handle, " << op->dtypes[0] << "* in, "
       << op->dtypes[1]
       << "* out, void* gain, void* bias, void* mean, void* variance, double eps)\n";
    lu.block_begin();
    {
        auto tensor_desc = cudnn_tensor_descriptor_from_shape(op->tensor_shape, "tensor_desc");
        lu << tensor_desc->get_code();
        // derived_param_desc
        lu << "cudnnTensorDescriptor_t derived_param_desc;\n";
        lu << "CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&derived_param_desc));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDeriveBNTensorDescriptor(derived_param_desc, tensor_desc, "
              "CUDNN_BATCHNORM_SPATIAL));\n";
        lu << "const float alpha = 1.0;\n";
        lu << "const float beta = 0.0;\n";
        lu << "CUDNN_SAFE_CALL(cudnnBatchNormalizationForwardInference(cudnn_handle,"
           << " CUDNN_BATCHNORM_SPATIAL,"
           << " &alpha,"
           << " &beta,"
           << " tensor_desc,"
           << " in,"
           << " tensor_desc,"
           << " out,"
           << " derived_param_desc,"
           << " gain,"     // gain
           << " bias,"     // bias
           << " mean,"     // mean
           << " variance," // variance
           << " eps));\n";

        lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(derived_param_desc));\n";
    }
    lu.block_end();
    return plu;
}

LanguageUnit_p cuda::BatchNorm::codegen_function_call()
{
    LanguageUnit_p plu(new LanguageUnit(codegen_function_name() + "_call"));
    auto& lu = *plu;
    lu << codegen_function_name() << "(global_cudnn_handle, " << op->arg_names[2] << ", " // in
       << op->out_names[0] << ", "                                                        // out
       << op->arg_names[0] << ", "                                                        // gain
       << op->arg_names[1] << ", "                                                        // bias
       << op->arg_names[3] << ", "                                                        // mean
       << op->arg_names[4] << ", " // variance
       << op->epsilon << ");\n";
    return plu;
}

LanguageUnit_p cuda::BatchNorm::codegen_dependency()
{
    create_ptr(LanguageUnit, _lu, codegen_function_name() + "_dep");
    auto& lu = *_lu;
    lu.require(header::cudnn);
    lu.require(declaration::global_cudnn_handle);
    return _lu;
}

cuda::CudaFunction_p cuda::BatchNorm::codegen(ir::Operator_p inter_op)
{
    BatchNorm_p cop(new BatchNorm(inter_op));
    NGRAPH_DEBUG << "Codegen for BatchNorm function:" << cop->codegen_function_name() << endl;
    return cop;
}