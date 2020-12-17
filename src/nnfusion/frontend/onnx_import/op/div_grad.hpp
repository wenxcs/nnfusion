//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "../util/reduce_grad.hpp"
#include "core/node.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateDivGradOp(const onnx::NodeProto& node_proto,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    // out = x / y, x_grad = out_grad / y, y_grad = - out_grad * x / y ** 2
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    NNFUSION_CHECK(input_indexes.size() == 3);

                    auto out_grad = input_indexes[0];
                    auto x = input_indexes[1];
                    auto x_shape = x.get_shape();
                    std::tie(x, out_grad) =
                        graph::numpy_broadcast(std::make_pair(x, out_grad), m_graph);
                    auto y = input_indexes[2];
                    auto y_shape = y.get_shape();
                    std::tie(y, out_grad) =
                        graph::numpy_broadcast(std::make_pair(y, out_grad), m_graph);

                    // x_grad
                    auto x_grad_op = std::make_shared<op::Divide>();
                    x_grad_op->set_name(node_proto.output(0));
                    auto x_grad_gnode = m_graph->add_node_and_edge(x_grad_op, {out_grad, y});
                    auto x_grad = nnfusion::frontend::onnx_import::reduce::reduce_grad(
                        GNodeIndex{x_grad_gnode}, x_shape, m_graph);

                    // y_grad
                    auto numerator_gnode =
                        m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {out_grad, x});
                    auto denominator_gnode =
                        m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {y, y});
                    auto frac_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Divide>(), {numerator_gnode, denominator_gnode});
                    auto y_grad_op = std::make_shared<op::Negative>();
                    y_grad_op->set_name(node_proto.output(1));
                    auto y_grad_gnode = m_graph->add_node_and_edge(y_grad_op, {frac_gnode});
                    auto y_grad = nnfusion::frontend::onnx_import::reduce::reduce_grad(
                        GNodeIndex{y_grad_gnode}, y_shape, m_graph);

                    return {{node_proto.output(0), x_grad}, {node_proto.output(1), y_grad}};
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion
