// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "nnfusion/engine/engine.h"
#include "nnfusion/engine/interpreter.h"
#include "nnfusion/engine/op.h"

namespace nnfusion
{
    namespace interpreter
    {
        class CodeGenerator : public IInterpreterPass
        {
        public:
            bool run(std::shared_ptr<InterpreterContext> ctx,
                     std::shared_ptr<TranslationUnit> tu) override;
        };
    }
}