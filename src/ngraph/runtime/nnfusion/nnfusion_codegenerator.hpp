// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_op.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_functiontranslator.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            class ICodeGeneratorPass
            {
            public:
                virtual bool run(std::shared_ptr<IntermediateOP>& inter_op) = 0;

                static bool
                    run_passes(const std::vector<std::shared_ptr<ICodeGeneratorPass>>& passes,
                               std::shared_ptr<IntermediateOP>& inter_op)
                {
                    bool rc = true;
                    for (const auto& pass : passes)
                    {
                        rc = pass->run(inter_op);
                        if (!rc)
                            break;
                    }
                    return rc;
                }
            };

            class CodeGeneratorContext
            {
            public:
                std::unordered_map<IntermediateOP*, std::string> fun_src_buffer;
                std::unordered_map<std::string, std::string> m_variable_name_map;
                std::map<std::string, size_t> m_name_index_map;
                /*std::unordered_map<std::shared_ptr<Function>, std::list<std::shared_ptr<Node>>>
                    m_function_ordered_ops;*/
            };

            class CodeGenerator
            {
            public:
                CodeGenerator();

                bool codegen(std::shared_ptr<TranslationUnitMap> inter_op);
                bool codegen(std::shared_ptr<IntermediateOP>& inter_op);
                bool codegen(
                    std::shared_ptr<std::vector<std::shared_ptr<IntermediateOP>>> inter_ops);

            private:
                std::shared_ptr<CodeGeneratorContext> default_ctx;
                std::vector<std::shared_ptr<ICodeGeneratorPass>> pass_manager;
            };
        }
    }
}