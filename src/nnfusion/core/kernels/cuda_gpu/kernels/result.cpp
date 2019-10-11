// Microsoft (c) 2019, NNFusion Team

#include "result.hpp"
#include "../cuda_cudnn.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Result::Result(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    enforce(ctx->inputs.size() == 1) << "Input size mismatches.";
    enforce(ctx->outputs.size() == 1) << "Output size mismatches.";

    std::stringstream tag;
    tag << "cuda_result";
    custom_tag = tag.str();
}

LanguageUnit_p cuda::Result::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
    auto& lu = *_lu;

    vector<string> params;
    params.push_back(m_context->inputs[0].get_type() + "* input0");
    params.push_back(m_context->outputs[0].get_type() + "** output0");
    lu << "void "
       << "(" << join(params, ", ") << ")";
    return _lu;
}

LanguageUnit_p cuda::Result::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    *_lu << "*output0 = input0;";
    return _lu;
}

LanguageUnit_p cuda::Result::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    _lu->require(header::cuda);
    _lu->require(macro::CUDA_SAFE_CALL);

    return _lu;
}

REGISTER_KERNEL_EMITTER("Result",                                                  // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_lib"), // attrs
                        cuda::Result)                                              // constructor