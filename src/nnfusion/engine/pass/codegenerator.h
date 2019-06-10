// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "nnfusion/engine/engine.h"
#include "nnfusion/engine/interpreter.h"
#include "nnfusion/engine/op.h"

namespace nnfusion
{
        class CodeGenerator : public IInterpreterPass
        {
        public:
            CodeGenerator(string DeviceStr)
            {
                device = DeviceStr;
            }
            bool run(std::shared_ptr<InterpreterContext> ctx,
                     std::shared_ptr<TranslationUnit> tu) override;
        private:
            string device = "cuda";
        };
}