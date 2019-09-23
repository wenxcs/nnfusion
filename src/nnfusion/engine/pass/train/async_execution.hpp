// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/op.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

namespace nnfusion
{
    class TrainningAsyncExecution : public IInterpreterPass
    {
    public:
        TrainningAsyncExecution(DeviceType dt = CUDA_GPU)
            : m_device(dt)
        {
        }
        bool run(std::shared_ptr<InterpreterContext> ctx,
                 std::shared_ptr<TranslationUnit> tu) override;

    private:
        DeviceType m_device;
    };
}