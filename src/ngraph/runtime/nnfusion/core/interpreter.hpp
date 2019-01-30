// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "common.hpp"
#include "op.hpp"

namespace nnfusion
{
    class FunctionTranslatorContext;
    class TranslationUnit;

    class IFunctionTranslatorPass
    {
    public:
        virtual bool run(shared_ptr<FunctionTranslatorContext> ctx,
                         shared_ptr<TranslationUnit> tu,
                         shared_ptr<ngraph::Function> function) = 0;

        static bool run_passes(const vector<shared_ptr<IFunctionTranslatorPass>>& passes,
                               shared_ptr<FunctionTranslatorContext> ctx,
                               shared_ptr<TranslationUnit> tu,
                               shared_ptr<ngraph::Function> function)
        {
            bool rc = true;
            for (auto& pass : passes)
            {
                rc = pass->run(ctx, tu, function);
                if (!rc)
                    break;
            }
            return rc;
        }
    };

    class TranslationUnit
    {
    public:
        shared_ptr<ngraph::Function> m_function;
        shared_ptr<vector<ir::Operator_p>> inter_ops;
        shared_ptr<set<string>> input_names;
        shared_ptr<set<string>> output_names;
        shared_ptr<set<shared_ptr<ngraph::descriptor::Tensor>>> constants;
        vector<shared_ptr<ngraph::descriptor::Tensor>> arg;
        vector<shared_ptr<ngraph::descriptor::Tensor>> out;
        bool m_is_translated;
        size_t memory_pool_size;
        TranslationUnit()
            : inter_ops(new vector<ir::Operator_p>())
            , memory_pool_size(0)
            , m_is_translated(false)
            , input_names(new set<string>())
            , output_names(new set<string>())
            , constants(new set<shared_ptr<ngraph::descriptor::Tensor>>()){};
    };

    using TranslationUnitMap = map<shared_ptr<ngraph::Function>, shared_ptr<TranslationUnit>>;

    class FunctionTranslatorContext
    {
    public:
        bool m_is_translated;
        shared_ptr<ngraph::Function> m_function;
        // Store the function(model) and its corresponding Nodes.
        unordered_map<shared_ptr<Function>, list<shared_ptr<Node>>> m_function_ordered_ops;
        // (?)
        map<string, size_t> m_name_index_map;
        // Store Translated OP's
        unordered_map<Node*, Node*> m_node_function_map;
        unordered_map<shared_ptr<Node>, ir::Operator_p> m_node_inter_map;
        size_t m_offset;
        string m_function_name;
        unordered_map<string, size_t> m_tensor_memory_buffers;
        unordered_map<string, string> m_variable_name_map;
    };

    // This is to translate NGraph::Function to NNFusion::IntermediateOP
    class FunctionTranslator
    {
        friend class nnfusion_Backend;

    public:
        FunctionTranslator();
        FunctionTranslator(shared_ptr<vector<shared_ptr<IFunctionTranslatorPass>>> m_passes,
                           shared_ptr<FunctionTranslatorContext> ctx);
        ~FunctionTranslator(){};

        shared_ptr<TranslationUnitMap> translate(shared_ptr<ngraph::Function> function);

        static const size_t s_memory_pool_alignment;

    private:
        shared_ptr<FunctionTranslatorContext> m_trans_ctx;
        shared_ptr<vector<shared_ptr<IFunctionTranslatorPass>>> m_passes;

        ir::Operator_p translate_node(shared_ptr<Node> node);
    };
}