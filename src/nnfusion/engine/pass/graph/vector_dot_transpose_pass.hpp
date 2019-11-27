// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "ngraph/op/constant.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

DEFINE_bool(ftranspose_vecdot, false, "Enable vectdot transpose.");

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class VectorDotTransposePass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override
                {
                    bool using_pass = FLAGS_ftranspose_vecdot;
                    if (!using_pass)
                        return true;

                    LOG(INFO) << "Vector Dot Transpose Pass starts up for Graph: "
                              << graph->get_name();

                    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
                    std::set<std::shared_ptr<GNode>> const_nodes = {};
                    std::set<std::shared_ptr<GNode>> down_streams = {};

                    // Find nodes with all constant upstream nodes
                    for (auto& it : nodes)
                        if (it->get_op_type() == "Dot")
                        {
                            auto dot = std::dynamic_pointer_cast<ngraph::op::Dot>(it->get_op_ptr());
                            CHECK_NOT_NULLPTR(dot);
                            if (dot->get_transpose_B())
                                continue;
                            std::vector<std::shared_ptr<nnfusion::graph::Edge>> in_edges;
                            for (auto& edge : it->get_in_edges())
                                if (!edge->is_control_edge())
                                    in_edges.push_back(edge);
                            CHECK(in_edges.size() == 2);
                            std::sort(in_edges.begin(),
                                      in_edges.end(),
                                      [](const std::shared_ptr<nnfusion::graph::Edge>& a,
                                         const std::shared_ptr<nnfusion::graph::Edge>& b) {
                                          return (size_t)a->get_dst_input() <
                                                 (size_t)b->get_dst_input(); // put -1 to the end
                                      });
                            auto p_const = std::dynamic_pointer_cast<ngraph::op::Constant>(
                                in_edges[1]->get_src()->get_op_ptr());
                            if (!in_edges[1]->get_src()->is_constant() || p_const->is_parameter())
                                continue;
                            CHECK(in_edges[0]->get_src()->get_output_size() == 1)
                                << in_edges[0]->get_src()->get_op_type()
                                << "must has exactly one output.";
                            auto input0_shape =
                                in_edges[0]->get_src()->get_outputs().at(0)->get_shape();
                            if (input0_shape.size() != 2 || input0_shape[0] != 1)
                                continue;

                            auto output_tensor = in_edges[1]->get_src()->get_outputs().at(0);
                            size_t dtype_size = output_tensor->get_element_type().size();
                            if (dtype_size != 4)
                                continue;
                            Shape new_shape = {output_tensor->get_shape()[1],
                                               output_tensor->get_shape()[0]};
                            std::vector<int> values(new_shape[0] * new_shape[1]);
                            for (int i = 0; i < new_shape[0]; ++i)
                                for (int j = 0; j < new_shape[1]; ++j)
                                    values[i * new_shape[1] + j] =
                                        ((int*)p_const->get_data_ptr())[i + j * new_shape[0]];

                            dot->get_transpose_B() = true;
                            CHECK(output_tensor->get_shape().size() == 2);
                            auto new_constant = std::make_shared<ngraph::op::Constant>(
                                output_tensor->get_element_type(), new_shape, values.data());
                            in_edges[1]->get_src()->reset_op_ptr(new_constant);
                        }

                    LOG(INFO) << "";
                    LOG(INFO) << "Vector Dot Transpose Pass ends up for Graph: "
                              << graph->get_name();
                    LOG(INFO) << "";
                    return true;
                }
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
