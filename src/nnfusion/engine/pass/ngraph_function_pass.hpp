// Microsoft (c) 2019, Wenxiang Hu
#pragma once
#include "nnfusion/engine/engine.hpp"
#include "nnfusion/engine/interpreter.hpp"

namespace nnfusion
{
    namespace interpreter
    {
        class NgraphFunctionPass : public IInterpreterPass
        {
        public:
            bool run(std::shared_ptr<InterpreterContext> ctx,
                     std::shared_ptr<TranslationUnit> tu) override;
        };
    }
}
