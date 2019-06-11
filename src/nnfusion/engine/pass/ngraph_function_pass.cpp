// Microsoft (c) 2019, Wenxiang Hu
#include "ngraph_function_pass.hpp"

using namespace nnfusion::interpreter;

bool NgraphFunctionPass::run(std::shared_ptr<InterpreterContext> ctx,
                             std::shared_ptr<TranslationUnit> tu)
{
    std::shared_ptr<ngraph::Function> function = tu->m_function;
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::AssignLayout<descriptor::layout::DenseTensorLayout>>();
    pass_manager.register_pass<ngraph::pass::Liveness>();
    pass_manager.register_pass<ngraph::pass::MemoryLayout>(64);
    pass_manager.run_passes(function);
    for (std::shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        ctx->m_function_ordered_ops.emplace(current_function, current_function->get_ordered_ops());
    }
    return true;
}