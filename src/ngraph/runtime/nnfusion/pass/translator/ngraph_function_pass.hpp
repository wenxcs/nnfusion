// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_functiontranslator.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            namespace translator
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
    }
}
