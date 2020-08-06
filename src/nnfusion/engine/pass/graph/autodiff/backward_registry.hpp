// Microsoft (c) 2020, NNFusion Team

#pragma once

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion::graph;
inline GNodeIndex get_node_input(std::shared_ptr<GNode> node, size_t i)
{
    auto in_edge = node->get_in_edge(i);
    NNFUSION_CHECK(in_edge);
    return GNodeIndex{in_edge->get_src(), in_edge->get_src_output()};
}

inline GNodeIndex get_node_output(std::shared_ptr<GNode> node, size_t i)
{
    return GNodeIndex(node, i);
}

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            namespace autodiff
            {
                ///\todo forward_node should be const, std::shared_ptr<const GNode> forward_node
                using BackwardTranslator =
                    std::function<GNodeIndexVector(std::shared_ptr<GNode> forward_node,
                                                   const GNodeIndexVector& outputs_grad,
                                                   std::shared_ptr<nnfusion::graph::Graph> graph)>;
                class BackwardTranslatorConfig
                {
                public:
                    BackwardTranslatorConfig& name(const string& name)
                    {
                        m_name = name;
                        return *this;
                    }
                    BackwardTranslatorConfig& translator(const BackwardTranslator& translator)
                    {
                        m_translator = translator;
                        return *this;
                    }

                    BackwardTranslator m_translator;
                    string m_name;
                };

                class DiffEngine
                {
                public:
                    static const GNodeIndex EMPTY_GNODE_INDEX;

                    DiffEngine() = delete;
                    DiffEngine(const DiffEngine& diff_engine) = delete;
                    DiffEngine& operator=(const DiffEngine& diff_engine) = delete;
                    DiffEngine(std::shared_ptr<Graph> graph)
                        : m_graph(graph)
                    {
                    }
                    void add_delta(const GNodeIndex& x, const GNodeIndex& grad);
                    void differentiate_graph(const GNodeIndexVector& outputs,
                                             const GNodeIndexVector& outputs_grad);
                    static std::unordered_map<string, BackwardTranslatorConfig>& Registry()
                    {
                        static std::unordered_map<string, BackwardTranslatorConfig>
                            m_backward_registry;
                        return m_backward_registry;
                    }

                private:
                    std::shared_ptr<Graph> m_graph;
                    GNodeIndexVector m_outputs;
                    std::map<std::shared_ptr<GNode>, GNodeIndexVector> m_adjoint_map;
                };

                inline BackwardTranslatorConfig& build_backward_config(const string& name)
                {
                    auto& config = DiffEngine::Registry()[name];
                    config.name(name);
                    return config;
                }
            }
        }
    }
}

#define REGISTER_BACKWARD_TRANSLATOR(op_name)                                                      \
    static nnfusion::pass::graph::autodiff::BackwardTranslatorConfig                               \
        __register_backward_translator_##op_name =                                                 \
            nnfusion::pass::graph::autodiff::build_backward_config(#op_name)
