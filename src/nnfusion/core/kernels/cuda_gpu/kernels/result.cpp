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
    //Inplace
    result = static_pointer_cast<ngraph::op::Result>(ctx->node);
    is_inplace = false;
    //Inplace
    auto annotations = result->get_op_annotations();
    if (annotations && annotations->get_in_place_oi_pairs().size() > 0)
    {
        is_inplace = true;
    }
}

LanguageUnit_p cuda::Result::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    auto dst = m_context->outputs[0];
    auto src = m_context->inputs[0];
    lu << dst.get_type() << "* " << dst.get_name() << " = output0;\n";
    lu << src.get_type() << "* " << src.get_name() << " = input0;\n";

    if (!is_inplace)
    {
        emit_memcpyDtD(lu, dst, src);
    }

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