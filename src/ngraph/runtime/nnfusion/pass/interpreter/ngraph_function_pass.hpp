// Microsoft (c) 2019, Wenxiang Hu
#pragma once
#include "../../core/common.hpp"
#include "../../core/interpreter.hpp"

namespace nnfusion
{
    namespace interpreter
    {
        class NgraphFunctionPass : public IFunctionTranslatorPass
        {
        public:
            bool run(std::shared_ptr<FunctionTranslatorContext> ctx,
                     std::shared_ptr<TranslationUnit> tu,
                     std::shared_ptr<Function> function) override;
        };
    }
}
