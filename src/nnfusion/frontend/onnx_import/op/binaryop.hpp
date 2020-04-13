//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <memory>

#include "../core/node.hpp"
#include "../util/broadcasting.hpp"
#include "../util/util.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                template <typename T>
                NamedNodeVector
                    TranslateLegacyBinaryOp(const onnx::NodeProto& node_proto,
                                            const NodeMap& all_ng_nodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto lhs_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    auto rhs_gnode = GetInputNode(all_ng_nodes, node_proto, 1);
                    Node node(node_proto);
                    auto axis = node.get_attribute_value<int64_t>("axis", 0);

                    std::tie(lhs_gnode, rhs_gnode) = legacy_style_broadcast_for_binary_operation(
                        std::make_pair(lhs_gnode, rhs_gnode), axis, m_graph);

                    auto op = std::make_shared<T>();
                    NNFUSION_CHECK(node_proto.output_size() == 1)
                        << "Binary op should only has one output.";
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, {lhs_gnode, rhs_gnode});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

                template <typename T>
                NamedNodeVector TranslateBinaryOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto lhs_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    auto rhs_gnode = GetInputNode(all_ng_nodes, node_proto, 1);

                    std::tie(lhs_gnode, rhs_gnode) =
                        graph::numpy_broadcast(std::make_pair(lhs_gnode, rhs_gnode), m_graph);

                    auto op = std::make_shared<T>();
                    NNFUSION_CHECK(node_proto.output_size() == 1)
                        << "Binary op should only has one output.";
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, {lhs_gnode, rhs_gnode});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

            } // namespace set_1

            namespace set_7
            {
                template <typename T>
                NamedNodeVector TranslateBinaryOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto lhs_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    auto rhs_gnode = GetInputNode(all_ng_nodes, node_proto, 1);

                    std::tie(lhs_gnode, rhs_gnode) =
                        graph::numpy_broadcast(std::make_pair(lhs_gnode, rhs_gnode), m_graph);

                    auto op = std::make_shared<T>();
                    NNFUSION_CHECK(node_proto.output_size() == 1)
                        << "Binary op should only has one output.";
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, {lhs_gnode, rhs_gnode});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

            } // namespace set_7
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace ngraph
