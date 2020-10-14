// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "reduce_fusion_pass.hpp"
#include <queue>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

DEFINE_bool(freduce_fusion, false, "Enable reduce-range based kernel fusion.");

namespace
{
    struct TaggedNode
    {
        TaggedNode()
            : node(nullptr)
            , ready_inputs(0)
            , visited(false)
        {
        }

        std::shared_ptr<GNode> node;
        size_t ready_inputs;
        bool visited;
    };
}

class ReduceFusionOptimizer
{
public:
    ReduceFusionOptimizer(std::shared_ptr<Graph> g)
        : m_graph(g)
    {
        m_nodes.resize(m_graph->get_max_node_id());
    };

    bool Optimize()
    {
        std::queue<size_t> ready;
        for (auto node : m_graph->get_ordered_ops())
        {
            size_t id = node->get_id();
            m_nodes[id] = std::make_shared<TaggedNode>();
            m_nodes[id]->node = node;
            if (!(m_nodes[id]->visited) &&
                (m_nodes[id]->ready_inputs == node->get_in_edges().size()))
            {
                ready.push(id);
            }
        }

        while (!ready.empty())
        {
            size_t node_id = ready.front();
            ready.pop();

            node_id = ReduceFusion(node_id);
            auto tn = m_nodes[node_id];
            tn->visited = true;
            for (auto edge : tn->node->get_out_edges())
            {
                auto dst = m_nodes[edge->get_dst()->get_id()];
                // node that will not be computed
                if (!dst)
                    continue;
                dst->ready_inputs++;
                NNFUSION_CHECK(!(dst->visited) || dst->node->get_op_type() == "Matched_Pattern");
                if (dst->ready_inputs >= dst->node->get_in_edges().size())
                {
                    NNFUSION_CHECK(dst->ready_inputs == dst->node->get_in_edges().size());
                    ready.push(dst->node->get_id());
                }
            }
        }
    }

private:
    size_t ReduceFusion(size_t id)
    {
        std::unordered_set<std::shared_ptr<GNode>> fusable;
        std::shared_ptr<GNode> tn = m_nodes[id]->node;

        auto op = tn->get_op_ptr();
        if (op->get_shared_memory().empty())
        {
            return id;
        }
    }
    std::shared_ptr<Graph> m_graph;
    std::vector<std::shared_ptr<TaggedNode>> m_nodes;
};

bool ReduceFusionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    if (FLAGS_freduce_fusion)
    {
        ReduceFusionOptimizer optimizer(graph);
        return optimizer.Optimize();
    }
    return true;
}