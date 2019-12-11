// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
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

                    std::vector<std::shared_ptr<GNode>> tail_op;
                    std::vector<int> tail_op_idx;

                    auto is_transpose = [](const std::shared_ptr<GNode>& gnode) -> bool {
                        return std::dynamic_pointer_cast<op::Reshape>(gnode->get_op_ptr())
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
                    {
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
                    }

                    for (int i = 0; i < tail_op.size(); ++i)
                    {
                        std::vector<std::shared_ptr<GNode>> chain;
                        auto node = tail_op[i]->get_in_edge(tail_op_idx[i])->get_src();
                        CHECK_NOT_NULLPTR(node);
                        CHECK(node->get_op_type() == "Reshape");
                        chain.push_back(node);
                        while (true)
                        {
                            node = node->get_in_edge(0)->get_src();
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
                        for (int i = 0; i < node->get_output_shape(0).size(); ++i)
                            order.push_back(i);
                        for (int i = chain.size() - 1; i >= 0; --i)
                        {
                            auto chord =
                                std::dynamic_pointer_cast<op::Reshape>(chain[i]->get_op_ptr())
                                    ->get_input_order();
                            CHECK(order.size() == chord.size());
                            mirror.resize(order.size());
                            for (int i = 0; i < chord.size(); ++i)
                                mirror[i] = order[chord[i]];
                            order = std::move(mirror);

                            // for (auto &it: chord) printf("%d ", (int)it); puts("");
                        }
                        auto top_shape = node->get_output_shape(0), out_shape = top_shape;
                        CHECK(top_shape.size() == order.size());
                        for (int i = 0; i < top_shape.size(); ++i)
                        {
                            out_shape[i] = top_shape[order[i]];
                        }
                        auto reshape_op = std::make_shared<op::Reshape>(order, out_shape);

                        auto reshape_gnode =
                            graph->add_node_and_edge(reshape_op, GNodeVector({node}));
                        graph->add_edge(reshape_gnode, 0, tail_op[i], tail_op_idx[i]);

                        graph->remove_edge(tail_op[i]->get_in_edge(tail_op_idx[i]));

                        // for (auto &it: order) printf("%d ", (int)it); puts("");
                        // printf("%s (%d) => %zd\n", tail_op[i]->get_op_type().c_str(), tail_op_idx[i], chain.size());
                    }

                    return true;
                }
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
