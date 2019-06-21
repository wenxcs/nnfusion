// Microsoft (c) 2019, Wenxiang Hu
#include "device_dispatcher.hpp"

using namespace nnfusion;

bool DefaultDeviceDispatcher::run(std::shared_ptr<InterpreterContext> ctx,
                                  std::shared_ptr<TranslationUnit> tu)
{
    auto& p = tu->program;
    DeviceType dt = default_device;
    /* for debug purpose
    switch(default_device)
    {
        case GENERIC_CPU:
        LOG_INFO << "GENERIC_CPU";
        break;
        case  ROCM_GPU:
        LOG_INFO << "ROCM_GPU";
        break;
        case CUDA_GPU:
        LOG_INFO << "CUDA_GPU";
    }
    */
    for (auto iterator = p.entry; iterator != nullptr; iterator = iterator->next)
    {
        for (auto ins : *iterator)
        {
            ins->Tag().Set<DeviceType>("Device", move(dt));
        }
    }
    return true;
}