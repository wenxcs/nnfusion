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
                NamedNodeVector
                    TranslateBatchNormOp(const onnx::NodeProto& node_proto,
                                         const NodeMap& all_ng_nodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto x_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    auto scale_gnode = GetInputNode(all_ng_nodes, node_proto, 1);
                    auto bias_gnode = GetInputNode(all_ng_nodes, node_proto, 2);

                    std::shared_ptr<graph::GNode> mean_gnode{nullptr};
                    std::shared_ptr<graph::GNode> var_gnode{nullptr};

                    Node node(node_proto);
                    std::int64_t is_test{node.get_attribute_value<std::int64_t>("is_test", 1)};
                    std::int64_t spatial{node.get_attribute_value<std::int64_t>("spatial", 1)};
                    double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};

                    // TODO: Implement learning mode support
                    // float momentum{node.get_attribute_value<float>("momentum", 0.9f)};
                    NNFUSION_CHECK(is_test == 1) << "only 'is_test' mode is supported.";
                    NNFUSION_CHECK(spatial == 1) << "only 'spatial' mode is supported.";

                    std::shared_ptr<graph::GNode> gnode;
                    if (node_proto.input_size() >= 5)
                    {
                        mean_gnode = GetInputNode(all_ng_nodes, node_proto, 3);
                        var_gnode = GetInputNode(all_ng_nodes, node_proto, 4);
                        auto op = std::make_shared<op::BatchNormInference>(epsilon);
                        op->set_name(node_proto.output(0));
                        gnode = m_graph->add_node_and_edge(
                            op, {scale_gnode, bias_gnode, x_gnode, mean_gnode, var_gnode});
                    }
                    else
                    {
                        auto op = std::make_shared<op::BatchNormTraining>(epsilon);
                        op->set_name(node_proto.output(0));
                        gnode = m_graph->add_node_and_edge(op, {scale_gnode, bias_gnode, x_gnode});
                    }

                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion
