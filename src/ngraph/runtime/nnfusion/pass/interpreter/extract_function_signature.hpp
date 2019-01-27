// Microsoft (c) 2019, Wenxiang Hu
#pragma once
#include "../../core/common.hpp"
#include "../../core/interpreter.hpp"

namespace nnfusion
{
    namespace interpreter
    {
        class ExtractFunctionSignature : public IFunctionTranslatorPass
        {
        public:
            bool extract_result(std::shared_ptr<TranslationUnit> tu,
                                std::shared_ptr<ngraph::Function> function);

            bool extract_constants(std::shared_ptr<FunctionTranslatorContext> ctx,
                                   std::shared_ptr<TranslationUnit> tu,
                                   std::shared_ptr<ngraph::Function> function);

            void propagate_in_place_input(std::shared_ptr<FunctionTranslatorContext> ctx,
                                          ngraph::descriptor::Output* output,
                                          std::string input_name);

            void propagate_in_place_output(std::shared_ptr<FunctionTranslatorContext> ctx,
                                           ngraph::descriptor::Output* res_src_output,
                                           std::string output_name);

            bool extract_args(std::shared_ptr<FunctionTranslatorContext> ctx,
                              std::shared_ptr<TranslationUnit> tu,
                              std::shared_ptr<ngraph::Function> function);

            bool extract_output(std::shared_ptr<FunctionTranslatorContext> ctx,
                                std::shared_ptr<TranslationUnit> tu,
                                std::shared_ptr<ngraph::Function> function);

            bool run(std::shared_ptr<FunctionTranslatorContext> ctx,
                     std::shared_ptr<TranslationUnit> tu,
                     std::shared_ptr<ngraph::Function> function) override;
        };
    }
}
