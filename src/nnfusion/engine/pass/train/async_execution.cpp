// Microsoft (c) 2019, NNFusion Team
#include "async_execution.hpp"

using namespace nnfusion;

bool TrainningAsyncExecution::run(std::shared_ptr<InterpreterContext> ctx,
                                  std::shared_ptr<TranslationUnit> tu)
{
    // Four streams in use:
    // default; d2h; h2d; allreduce;
    // allreduce waits for default stream reduce_start;
    //
    bool allreduce_enable = getenv("NNFUSION_ENABLE_ALLREDUCE")
                                ? bool(atoi(getenv("NNFUSION_ENABLE_ALLREDUCE")))
                                : false;
    if (!allreduce_enable)
        return true;

    auto& p = tu->program;
    for (auto iterator : p)
    {
        for (auto ins : *iterator)
        {
            // Detect those operators
            // 1). ApplyGradient
            //     |
            //      ----> (memcpy-DtoH)(AG)(memcpy-HtoD)
            //     memcpy will assign to D2H or H2D stream.
            //     AG will assign to default stream, and wait *ALL* gradient
            //     ready signal. (Or iteration end signal?)
            // 2). AllReduce
            //     |
            //      ----> (memcpy-HtoD)(AR)(memcpy-DtoH)
            //     memcpy will assign to H2D or H2D stream.
            //     AR will assign to its own stream.
            //     It will wait for Host Data ready signal.
            //     It will trigger Reduced signal.
            // For other memcpy operators, no stream assigned.
            auto node = ins->operatorDef();

            if (node->description() != "ApplyGradient" && node->description() != "AllReduce")
                continue;

            auto& emitted_kernels = (*ins)["Kernel_Selection_Result"]
                                        .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();
            auto emitter_iter = find_if(emitted_kernels.begin(),
                                        emitted_kernels.end(),
                                        [this](pair<DeviceType, KernelEmitter::Pointer>& i) {
                                            return i.first == this->m_device;
                                        });

            KernelEmitter::Pointer kernel = nullptr;

            if (emitter_iter == emitted_kernels.end() || emitter_iter->second == nullptr)
            {
                enforce(false)
                    << "AllRedeuce && ApplyGradient Kernel should be emitted before this pass.";
            }
            else
                kernel = emitter_iter->second;

            auto kernel_context = kernel->m_context;
            auto& async_info = kernel_context->async_info;

            if (node->description() == "AllReduce")
            {
                async_info.execution_stream.number = 1;
                async_info.execution_stream.name = kernel_context->output_names[0] + "_stream";
            }
            else if (node->description() == "ApplyGradient")
            {
                async_info.execution_stream.number = 2;
                async_info.execution_stream.name = kernel_context->input_names[1] + "_stream";
            }
        }
    }
    return true;
}