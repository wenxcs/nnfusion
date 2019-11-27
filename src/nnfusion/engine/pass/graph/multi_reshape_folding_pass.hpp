// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "ngraph/op/constant.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::graph;

DEFINE_bool(ffold_reshape_op, true, "Folding Reshape operators.");

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class MultiReshapeFoldingPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override
                {
                    bool using_pass = FLAGS_ffold_reshape_op;
                    if (!using_pass)
                        return true;

                    LOG(INFO) << "Multi Reshape Folding Pass starts up for Graph: "
                              << graph->get_name();

                    std::vector<std::shared_ptr<GNode>> tail_op;
                    std::vector<int> tail_op_idx;

                    auto is_transpose = [](const std::shared_ptr<GNode>& gnode) -> bool {
                        return std::dynamic_pointer_cast<ngraph::op::Reshape>(gnode->get_op_ptr())
                            ->get_is_transpose();
                    };

                    auto get_in_edge = [](const std::shared_ptr<GNode>& gnode,
                                          int idx) -> std::shared_ptr<nnfusion::graph::Edge> {
                        for (auto& it : gnode->get_in_edges())
                        {
                            if (it->get_dst_input() == idx)
                                return it;
                        }
                        return nullptr;
                    };

                    // Find tail nodes exactly after reshape node
                    for (auto& it : graph->get_nodes())
                        if (it->get_op_type() != "Reshape")
                        {
                            for (auto edge : it->get_in_edges())
                            {
                                if (edge->is_control_edge())
                                    continue;
                                if (edge->get_src()->get_op_type() == "Reshape")
                                {
                                    if (!is_transpose(edge->get_src()))
                                        continue;
                                    tail_op.push_back(it);
                                    tail_op_idx.push_back(edge->get_dst_input());
                                }
                            }
                        }

                    for (int i = 0; i < tail_op.size(); ++i)
                    {
                        std::vector<std::shared_ptr<GNode>> chain;
                        auto node = get_in_edge(tail_op[i], tail_op_idx[i])->get_src();
                        CHECK_NOT_NULLPTR(node);
                        CHECK(node->get_op_type() == "Reshape");
                        chain.push_back(node);
                        while (true)
                        {
                            node = get_in_edge(node, 0)->get_src();
                            if (node->get_op_type() == "Reshape" && is_transpose(node))
                                chain.push_back(node);
                            else
                                break;
                        }
                        if (chain.size() <= 1)
                            continue;
                        AxisVector order, mirror;
                        CHECK(node->get_output_size() == 1) << node->get_op_type()
                                                            << "must has exactly one output.";
                        for (int i = 0; i < node->get_outputs().at(0)->get_shape().size(); ++i)
                            order.push_back(i);
                        for (int i = chain.size() - 1; i >= 0; --i)
                        {
                            auto chord = std::dynamic_pointer_cast<ngraph::op::Reshape>(
                                             chain[i]->get_op_ptr())
                                             ->get_input_order();
                            CHECK(order.size() == chord.size());
                            mirror.resize(order.size());
                            for (int i = 0; i < chord.size(); ++i)
                                mirror[i] = order[chord[i]];
                            order = std::move(mirror);

                            // for (auto &it: chord) printf("%d ", (int)it); puts("");
                        }
                        auto top_shape = node->get_outputs().at(0)->get_shape(),
                             out_shape = top_shape;
                        CHECK(top_shape.size() == order.size());
                        for (int i = 0; i < top_shape.size(); ++i)
                        {
                            out_shape[i] = top_shape[order[i]];
                        }
                        auto reshape_op = std::make_shared<ngraph::op::Reshape>(
                            node->get_op_ptr(), order, out_shape);

                        auto reshape_node = graph->add_node(reshape_op);
                        graph->remove_edge(get_in_edge(tail_op[i], tail_op_idx[i]));
                        graph->add_edge(reshape_node, 0, tail_op[i], tail_op_idx[i]);
                        graph->add_edge(node, 0, reshape_node, 0);

                        // TODO : to be removed once gnode input is used
                        tail_op[i]->get_op_ptr()->get_inputs()[tail_op_idx[i]].replace_output(
                            reshape_node->get_op_ptr(), 0);
                        reshape_node->get_op_ptr()->get_inputs()[0].replace_output(
                            node->get_op_ptr(), 0);
                        // end TODO

                        // for (auto &it: order) printf("%d ", (int)it); puts("");
                        // printf("%s (%d) => %zd\n", tail_op[i]->get_op_type().c_str(), tail_op_idx[i], chain.size());
                    }

                    LOG(INFO) << "";
                    LOG(INFO) << "Multi Reshape Folding Pass ends up for Graph: "
                              << graph->get_name();
                    LOG(INFO) << "";
                    return true;
                }
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
