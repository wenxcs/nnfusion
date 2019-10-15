// Microsoft (c) 2019, NNFusion Team

#include "generate_result_op_pass.hpp"
#include "../gnode.hpp"
#include "../graph.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::graph::pass;

bool GenerateResultOpPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    std::vector<std::shared_ptr<GNode>> result_nodes;
    for (auto node : graph->get_outputs())
    {
        auto result_op = std::make_shared<ngraph::op::Result>(node->get_op_ptr());
        auto result_node = graph->add_node(result_op);
        graph->add_edge(node, 0, result_node, 0);
        result_nodes.emplace_back(result_node);
    }
    graph->set_outputs(result_nodes);
    return true;
}