// Microsoft (c) 2019, NNFusion Team

#include "anyop.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::AnyOP::AnyOP(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    std::stringstream tag;
    tag << "_AnyOP";
    custom_tag = tag.str();
}

LanguageUnit_p cuda::AnyOP::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    lu.block_begin();
    {
        lu << "// This function is left blank by purpose.\n";
    }
    lu.block_end();
    return _lu;
}

LanguageUnit_p cuda::AnyOP::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

// Register Pad kernel emitter

REGISTER_KERNEL_EMITTER("AnyOP",                                   //op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT), //attrs
                        cuda::AnyOP)                               // constructor