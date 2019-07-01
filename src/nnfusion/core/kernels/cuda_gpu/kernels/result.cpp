// Microsoft (c) 2019, NNFusion Team

#include "result.hpp"
#include "../cuda_cudnn.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Result::Result(shared_ptr<KernelContext> ctx)
    : CudaEmitter(ctx)
{
    enforce(ctx->inputs.size() == 1) << "Input size mismatches.";
    enforce(ctx->outputs.size() == 1) << "Output size mismatches.";

    std::stringstream tag;
    tag << "cuda_result";
    custom_tag = tag.str();
}

LanguageUnit_p cuda::Result::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    emit_memcpyDtD(lu, m_context->outputs[0], m_context->inputs[0]);

    return _lu;
}

void cuda::Result::set_launch_config()
{
}

LanguageUnit_p cuda::Result::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    _lu->require(header::cuda);
    _lu->require(macro::CUDA_SAFE_CALL);

    return _lu;
}

REGISTER_KERNEL_EMITTER("Result",                                                     // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::Result)                                                 // constructor