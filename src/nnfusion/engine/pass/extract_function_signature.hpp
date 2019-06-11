// Microsoft (c) 2019, Wenxiang Hu
#pragma once
#include "nnfusion/engine/engine.hpp"
#include "nnfusion/engine/interpreter.hpp"

namespace nnfusion
{
    namespace interpreter
    {
        class ExtractFunctionSignature : public IInterpreterPass
        {
        public:
            bool extract_result(std::shared_ptr<TranslationUnit> tu,
                                std::shared_ptr<ngraph::Function> function);

            bool extract_constants(std::shared_ptr<InterpreterContext> ctx,
                                   std::shared_ptr<TranslationUnit> tu,
                                   std::shared_ptr<ngraph::Function> function);

            void propagate_in_place_input(std::shared_ptr<InterpreterContext> ctx,
                                          ngraph::descriptor::Output* output,
                                          std::string input_name);

            void propagate_in_place_output(std::shared_ptr<InterpreterContext> ctx,
                                           ngraph::descriptor::Output* res_src_output,
                                           std::string output_name);

            bool extract_args(std::shared_ptr<InterpreterContext> ctx,
                              std::shared_ptr<TranslationUnit> tu,
                              std::shared_ptr<ngraph::Function> function);

            bool extract_output(std::shared_ptr<InterpreterContext> ctx,
                                std::shared_ptr<TranslationUnit> tu,
                                std::shared_ptr<ngraph::Function> function);

            bool run(std::shared_ptr<InterpreterContext> ctx,
                     std::shared_ptr<TranslationUnit> tu) override;
        };
    }
}
