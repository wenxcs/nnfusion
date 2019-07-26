// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/core/IR/instruction.hpp"
#include "op.hpp"

namespace nnfusion
{
    class InterpreterContext;
    class TranslationUnit;

    class IInterpreterPass
    {
    public:
        virtual bool run(shared_ptr<InterpreterContext> ctx, shared_ptr<TranslationUnit> tu) = 0;

        static bool run_passes(const vector<shared_ptr<IInterpreterPass>>& passes,
                               shared_ptr<InterpreterContext> ctx,
                               shared_ptr<TranslationUnit> tu)
        {
            bool rc = true;
            for (auto& pass : passes)
            {
                rc = pass->run(ctx, tu);
                if (!rc)
                    break;
            }
            return rc;
        }
    };

    class BasicBlock : public std::vector<ir::Instruction::Pointer>
    {
    public:
        using pointer = std::shared_ptr<BasicBlock>;
        using pointers = std::shared_ptr<vector<BasicBlock>>;
        pointer next, prior;
    };

    class Program : BasicBlock
    {
    public:
        using pointer = std::shared_ptr<Program>;
        BasicBlock::pointer entry, exit;
    };

    class TranslationUnit
    {
    public:
        using Pointer = shared_ptr<TranslationUnit>;
        shared_ptr<ngraph::Function> m_function;
        shared_ptr<graph::Graph> m_graph;
        shared_ptr<vector<ir::Operator_p>> inter_ops;
        shared_ptr<set<string>> input_names;
        shared_ptr<set<string>> output_names;
        shared_ptr<set<shared_ptr<ngraph::descriptor::Tensor>>> constants;
        vector<shared_ptr<ngraph::descriptor::Tensor>> arg;
        vector<shared_ptr<ngraph::descriptor::Tensor>> out;
        Program program;
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
    using GraphTranslationUnitMap = map<shared_ptr<graph::Graph>, shared_ptr<TranslationUnit>>;

    class InterpreterContext
    {
    public:
        bool m_is_translated;
        shared_ptr<ngraph::Function> m_function;
        shared_ptr<graph::Graph> m_graph;

        // Store the function(model) and its corresponding Nodes.
        unordered_map<shared_ptr<Function>, list<shared_ptr<Node>>> m_function_ordered_ops;
        // TODO: multi graphs?
        unordered_set<shared_ptr<graph::Graph>> m_graphs;
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
    class Interpreter
    {
        friend class nnfusion_Backend;

    public:
        Interpreter();
        Interpreter(shared_ptr<vector<shared_ptr<IInterpreterPass>>> m_passes,
                    shared_ptr<InterpreterContext> ctx);
        ~Interpreter(){};

        shared_ptr<TranslationUnitMap> translate(shared_ptr<ngraph::Function> function);
        shared_ptr<GraphTranslationUnitMap> translate(shared_ptr<graph::Graph> graph);

        bool translate(TranslationUnit::Pointer tu);

        static const size_t s_memory_pool_alignment;

        shared_ptr<InterpreterContext> m_trans_ctx;
        shared_ptr<vector<shared_ptr<IInterpreterPass>>> m_passes;

        ir::Operator_p translate_node(shared_ptr<Node> node);
    };

    using Interpreter_p = shared_ptr<Interpreter>;
}