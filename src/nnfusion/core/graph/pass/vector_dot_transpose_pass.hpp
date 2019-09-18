// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "ngraph/op/constant.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

namespace nnfusion
{
    namespace graph
    {
        namespace pass
        {
            class VectorDotTransposePass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override
                {
                    bool using_pass = getenv("NNFUSION_ENABLE_VECDOT_TRANPOSE")
                                          ? atoi(getenv("NNFUSION_ENABLE_VECDOT_TRANPOSE"))
                                          : 1;
                    if (!using_pass)
                        return true;

                    LOG_INFO << "Vector Dot Transpose Pass starts up for Graph: "
                             << graph->get_name();

                    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
                    std::set<std::shared_ptr<GNode>> const_nodes = {};
                    std::set<std::shared_ptr<GNode>> down_streams = {};

                    // Find nodes with all constant upstream nodes
                    for (auto& it : nodes)
                        if (it->get_op_ptr()->description() == "Dot")
                        {
                            auto dot = std::dynamic_pointer_cast<ngraph::op::Dot>(it->get_op_ptr());
                            assert(dot != nullptr);
                            if (dot->get_transpose_B())
                                continue;
                            std::vector<std::shared_ptr<nnfusion::graph::Edge>> inputs;
                            for (auto& edge : it->get_in_edges())
                                if (!edge->is_control_edge())
                                    inputs.push_back(edge);
                            assert(inputs.size() == 2);
                            std::sort(inputs.begin(),
                                      inputs.end(),
                                      [](const std::shared_ptr<nnfusion::graph::Edge>& a,
                                         const std::shared_ptr<nnfusion::graph::Edge>& b) {
                                          return (size_t)a->get_dst_input() <
                                                 (size_t)b->get_dst_input(); // put -1 to the end
                                      });
                            auto p_const = std::dynamic_pointer_cast<ngraph::op::Constant>(
                                inputs[1]->get_src()->get_op_ptr());
                            if (!inputs[1]->get_src()->is_constant() || p_const->is_parameter())
                                continue;
                            auto x = inputs[0]->get_src()->get_op_ptr()->get_shape();
                            if (x.size() != 2 || x[0] != 1)
                                continue;
                            auto y = std::dynamic_pointer_cast<ngraph::op::Constant>(
                                inputs[1]->get_src()->get_op_ptr());
                            size_t dtype_size = y->get_output_element_type(0).size();
                            if (dtype_size != 4)
                                continue;
                            Shape new_shape = {y->get_shape()[1], y->get_shape()[0]};
                            std::vector<int> values(new_shape[0] * new_shape[1]);
                            for (int i = 0; i < new_shape[0]; ++i)
                                for (int j = 0; j < new_shape[1]; ++j)
                                    values[i * new_shape[1] + j] =
                                        ((int*)y->get_data_ptr())[i + j * new_shape[0]];

                            dot->get_transpose_B() = true;
                            assert(y->get_shape().size() == 2);
                            auto new_constant = std::make_shared<ngraph::op::Constant>(
                                y->get_output_element_type(0), new_shape, values.data());
                            inputs[1]->get_src()->reset_op_ptr(new_constant);
                        }

                    LOG_INFO << "";
                    LOG_INFO << "Vector Dot Transpose Pass ends up for Graph: "
                             << graph->get_name();
                    LOG_INFO << "";
                    return true;
                }
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
