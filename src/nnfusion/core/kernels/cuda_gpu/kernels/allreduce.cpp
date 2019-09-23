// Microsoft (c) 2019, NNFusion Team
#include <iostream>
#include <stdexcept>
#include <stdio.h>

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class SuperScalerAllReduce : public KernelEmitter
            {
            public:
                SuperScalerAllReduce(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "SuperScaler")
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    auto data_size = m_context->inputs.begin()->get_size();

                    lu << "cudaEvent_t* applygrad_ready = (cudaEvent_t "
                          "*)malloc(sizeof(cudaEvent_t));\n";
                    lu << "cudaEventCreate(applygrad_ready);\n";
                    lu << "cudaStreamWaitEvent(applygradient_stream, *applygrad_ready, 0);\n";
                    lu << "typedef void(*call_back_t)(cudaStream_t*, cudaEvent_t*);\n";
                    lu << "call_back_t call_back = (call_back_t)[](cudaStream_t* "
                          "applygradient_stream, cudaEvent_t* applygrad_ready)->void {\n";
                    lu << "    cudaEventRecord(*applygrad_ready, *applygradient_stream);\n";
                    // todo: is here be any problem that destroy event right after triggered?
                    lu << "    cudaEventDestroy(*applygrad_ready);\n";
                    lu << "    free((void*)applygrad_ready);\n";
                    lu << "};\n";
                    lu << "super_scaler_all_reduce_device_async(input0, output0, " << data_size
                       << ", call_back, &applygradient_stream, applygrad_ready);\n";
                    return _lu;
                }

                LanguageUnit_p emit_dependency()
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(header::super_scaler); // This require nccl, mpi
                    _lu->require(declaration::allreduce_stream);
                    _lu->require(declaration::applygradient_stream);
                    return _lu;
                }
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("AllReduce",                               //op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT), //attrs
                        cuda::SuperScalerAllReduce)                // constructor