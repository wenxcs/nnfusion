// Microsoft (c) 2019, Wenxiang Hu
#include "device_dispatcher.hpp"

using namespace nnfusion;

bool DefaultDeviceDispatcher::run(std::shared_ptr<InterpreterContext> ctx,
                                  std::shared_ptr<TranslationUnit> tu)
{
    auto& p = tu->program;
    for (auto iterator = p.entry; iterator != nullptr; iterator = iterator->next)
    {
        for (auto ins : *iterator)
        {
            ins->Tag().Set<DeviceType>("Device", CUDA_GPU);
        }
    }
    return true;
}