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
                             std::shared_ptr<Function> function) override
                    {
                        ngraph::pass::Manager pass_manager;
                        pass_manager.register_pass<
                            ngraph::pass::AssignLayout<descriptor::layout::DenseTensorLayout>>();
                        pass_manager.register_pass<ngraph::pass::MemoryLayout>(64);
                        pass_manager.run_passes(function);
                        for (std::shared_ptr<Function> current_function :
                             pass_manager.get_state().get_functions())
                        {
                            ctx->m_function_ordered_ops.emplace(
                                current_function, current_function->get_ordered_ops());
                        }
                    }
                };
            }
        }
    }
}
