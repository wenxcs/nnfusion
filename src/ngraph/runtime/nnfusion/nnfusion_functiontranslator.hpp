// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_op.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            class IFunctionTranslatorPass
            {
            public:
                virtual bool run(std::shared_ptr<ngraph::Function>& funcion) = 0;

                static bool
                    run_passes(const std::vector<std::shared_ptr<IFunctionTranslatorPass>>& passes,
                               std::shared_ptr<ngraph::Function>& function)
                {
                    bool rc = true;
                    for (auto& pass : passes)
                    {
                        rc = pass->run(function);
                        if (!rc)
                            break;
                    }
                    return rc;
                }
            };

            class TranslationUnit
            {
            public:
                std::shared_ptr<std::vector<std::shared_ptr<IntermediateOP>>> inter_ops;
                std::vector<void*> m_inputs;
                std::vector<void*> m_outputs;
                bool m_is_translated;
                TranslationUnit()
                    : inter_ops(new std::vector<std::shared_ptr<IntermediateOP>>())
                    , m_is_translated(false){};
            };

            class FunctionTranslatorContext
            {
            public:
                bool m_is_translated;
                std::shared_ptr<ngraph::Function> m_function;
                // Store the function(model) and its corresponding Nodes.
                std::unordered_map<std::shared_ptr<Function>, std::list<std::shared_ptr<Node>>>
                    m_function_ordered_ops;
                // (?)
                std::map<std::string, size_t> m_name_index_map;
                // Store Translated OP's
                std::unordered_map<Node*, Node*> m_node_function_map;
                std::unordered_map<const Node*, std::shared_ptr<IntermediateOP>> m_node_inter_map;
                size_t m_offset;
                std::string m_function_name;
                std::unordered_map<std::string, size_t> m_tensor_memory_buffers;
                std::unordered_map<std::string, std::string> m_variable_name_map;
            };

            // This is to translate NGraph::Function to NNFusion::IntermediateOP
            class FunctionTranslator
            {
                friend class nnfusion_Backend;

            public:
                FunctionTranslator()
                    : m_trans_ctx(new FunctionTranslatorContext()){};
                ~FunctionTranslator(){};

                bool translate(const std::shared_ptr<ngraph::Function> function);

                static const size_t s_memory_pool_alignment;

                std::shared_ptr<TranslationUnit> get_TranslationUnit();

            private:
                std::shared_ptr<FunctionTranslatorContext> m_trans_ctx;
                std::vector<std::shared_ptr<IFunctionTranslatorPass>> m_passes;
                // std::shared_ptr<GPU_Backend::BackendContext> m_shared_context;

                void propagate_in_place_input(ngraph::descriptor::Output* output,
                                              std::string input_name);

                void propagate_in_place_output(ngraph::descriptor::Output* res_src_output,
                                               std::string output_name);

                bool translate_node(TRANS_ARGS);
            };
        }
    }
}