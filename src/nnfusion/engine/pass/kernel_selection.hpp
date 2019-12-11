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
            profiling_best(shared_ptr<graph::GNode> gnode,
                           DeviceType devtype,
                           nnfusion::profiler::IProfilingRuntime::Pointer runtime);
    };

    class DefaultKernelSelector : public IInterpreterPass
    {
    public:
        bool run(std::shared_ptr<InterpreterContext> ctx,
                 std::shared_ptr<TranslationUnit> tu) override;
        pair<DeviceType, nnfusion::kernels::KernelEmitter::Pointer>
            pick_first(shared_ptr<graph::GNode> gnode, DeviceType devtype);
        pair<DeviceType, nnfusion::kernels::KernelEmitter::Pointer>
            pick_first_rocm(shared_ptr<graph::GNode> gnode);
    };
}