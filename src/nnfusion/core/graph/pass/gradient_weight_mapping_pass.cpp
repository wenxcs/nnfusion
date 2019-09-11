// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "gradient_weight_mapping_pass.hpp"
#include "../gnode.hpp"
#include "../graph.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::graph::pass;

bool GradientWeightMappingPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    std::vector<std::shared_ptr<GNode>> result_nodes;

    auto const_nodes = graph->get_const_nodes();

    for (auto node : graph->get_outputs())
    {
        std::shared_ptr<GNode> update_node = node;
        if ((*node)["Alias"].is_valid())
        {
            std::string alias = (*node)["Alias"].as<std::string>();
            std::string const_name = alias.substr(alias.find_first_of('/') + 1);
            auto gradient_shape = node->get_op_ptr()->get_output_shape(0);
            std::shared_ptr<GNode> weight_node = nullptr;
            for (auto const_node : const_nodes)
            {
                if (const_node->get_name() == const_name &&
                    const_node->get_op_ptr()->get_output_shape(0) == gradient_shape)
                {
                    weight_node = const_node;
                    break;
                }
            }
            if (weight_node != nullptr)
            {
                ngraph::op::OpConfig::any myConfig;
                myConfig["learning_rate"] = 0.001;

                auto apply_gradient_op = std::make_shared<ngraph::op::GenericOp>(
                    "apply_gradient_" + const_name,
                    "ApplyGradient",
                    std::vector<std::shared_ptr<Node>>(
                        {weight_node->get_op_ptr(), node->get_op_ptr()}),
                    myConfig);
                auto apply_gradient_node = graph->add_node(apply_gradient_op);
                graph->add_edge(weight_node, 0, apply_gradient_node, 0);
                graph->add_edge(node, 0, apply_gradient_node, 1);
                update_node = apply_gradient_node;
            }
        }
        // TODO: remove result op for gradient op
        auto result_op = std::make_shared<ngraph::op::Result>(update_node->get_op_ptr());
        auto result_node = graph->add_node(result_op);
        graph->add_edge(update_node, 0, result_node, 0);
        result_nodes.emplace_back(result_node);
    }
    graph->set_outputs(result_nodes);
    return true;
}