//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "core/node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateTanhGradOp(const onnx::NodeProto& node_proto,
                                                    const NodeMap& all_ng_nodes,
                                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    // y = tanh(x), x_grad = y_grad * (1 - x**2)
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    NNFUSION_CHECK(input_indexes.size() == 2);

                    auto x = input_indexes[0];
                    auto y_grad = input_indexes[1];

                    // x_grad
                    auto square_x =
                        m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {x, x});
                    auto one_op = std::make_shared<op::Constant>(
                        element::f32, x.get_shape(), std::vector<float>{1.0});
                    auto one_gnode =
                        m_graph->add_node_and_edge(one_op, nnfusion::graph::GNodeVector({}));
                    auto tanh_grad = m_graph->add_node_and_edge(std::make_shared<op::Subtract>(),
                                                                {one_gnode, square_x});
                    auto x_grad_op = std::make_shared<op::Multiply>();
                    x_grad_op->set_name(node_proto.output(0));
                    auto x_grad =
                        m_graph->add_node_and_edge(x_grad_op, {y_grad, GNodeIndex{tanh_grad}});

                    return {{node_proto.output(0), x_grad}};
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion
