// Microsoft (c) 2019, NNFUSION TEAM
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/op.hpp"

namespace nnfusion
{
    namespace pass
    {
        class HostTensorAllocation : public IInterpreterPass
        {
        public:
            HostTensorAllocation(DeviceType dt = CUDA_GPU)
                : m_device(dt)
            {
            }
            bool run(std::shared_ptr<InterpreterContext> ctx,
                     std::shared_ptr<TranslationUnit> tu) override;

        private:
            DeviceType m_device;
        };
    }
}