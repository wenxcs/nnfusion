// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/op.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

namespace nnfusion
{
    class ProfilingBasedKernelSelector : public IInterpreterPass
    {
    public:
        bool run(std::shared_ptr<InterpreterContext> ctx,
                 std::shared_ptr<TranslationUnit> tu) override;

        pair<DeviceType, nnfusion::kernels::KernelEmitter::Pointer>
            profiling_best(shared_ptr<ngraph::Node> node,
                           DeviceType devtype,
                           nnfusion::profiler::IProfilingRuntime::Pointer runtime);
    };
}