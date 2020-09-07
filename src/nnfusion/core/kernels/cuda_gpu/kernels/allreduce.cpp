// Microsoft (c) 2019, NNFusion Team
#include <iostream>
#include <stdexcept>
#include <stdio.h>

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class SuperScalerAllReduce : public KernelEmitter
            {
            public:
                string tensor_name;
                SuperScalerAllReduce(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "SuperScaler")
                {
                    tensor_name = ctx->output_names.front();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    auto data_size = m_context->inputs.front()->size(false);
                    auto code = nnfusion::op::create_code_from_template(
                        R"(assert(input0==output0);
auto call_sc_reduce = std::bind(sc_allreduce, "@tensorname@", input0, @dsize@);
auto thread_func = [&](){
    call_sc_reduce();
};
thread_func();
)",
                        {{"dsize", data_size}, {"tensorname", tensor_name}});
                    // allreduce and applygradient use the same stream.
                    lu << code;
                    return _lu;
                }

                LanguageUnit_p emit_dependency()
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(header::superscaler);
                    return _lu;
                }
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("AllReduce",                                           //op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Priority(2), //attrs
                        cuda::SuperScalerAllReduce)                            // constructor
