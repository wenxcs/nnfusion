// Microsoft (c) 2019, NNFusion Team

#include "batch_norm.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::BatchNorm::BatchNorm(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    // nnfusion::op::BatchNormInferece <-> nnfusion::ir::BatchNorm
    auto bn_op = static_pointer_cast<nnfusion::op::BatchNormInference>(ctx->gnode->get_op_ptr());
    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());
    // <todo> need to check the index
    tensor_shape = nnfusion::Shape(ctx->inputs[2]->get_shape());
    param_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    epsilon = bn_op->get_eps_value();

    std::stringstream tag;
    tag << "cudnn_batch_norm"
        << "_dtype_" << dtype.c_type_string() << "_i_" << join(tensor_shape, "_") << "_i_"
        << join(param_shape, "_") << "_" << ctx->outputs[0]->get_name();
    custom_tag = tag.str();
}

LanguageUnit_p cuda::BatchNorm::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    auto tensor_desc = cudnn_tensor_descriptor_from_shape(tensor_shape, "tensor_desc");
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
       << " input2,"
       << " tensor_desc,"
       << " output0,"
       << " derived_param_desc,"
       << " input0," // gain
       << " input1," // bias
       << " input3," // mean
       << " input4," // variance
       << epsilon << "));\n";

    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(derived_param_desc));\n";
    return _lu;
}

LanguageUnit_p cuda::BatchNorm::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cudnn);
    _lu->require(header::stdexcept);
    _lu->require(header::sstream);
    _lu->require(macro::CUDNN_SAFE_CALL);
    //_lu->require(declaration::cudnn_handle);
    return _lu;
}

LanguageUnit_p cuda::BatchNorm::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
    auto& lu = *_lu;

    vector<string> params;
    for (size_t i = 0; i < m_context->inputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
        ss << "input" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
        ss << "output" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->tensors.size(); i++)
    {
        stringstream ss;
        ss << m_context->tensors[i]->get_element_type().c_type_string() << "* ";
        // defult name is: "persit0", "persist1" ...
        ss << m_context->tensors[i]->get_name();
        params.push_back(ss.str());
    }

    lu << "void "
       << "(cudnnHandle_t cudnn_handle, " << join(params, ", ") << ")";
    return _lu;
}

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("BatchNormInference", // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cudnn").Priority(2), // attrs
                        cuda::BatchNorm) // constructor
