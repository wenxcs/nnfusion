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
                NamedNodeVector TranslateConcatOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnodes = GetAllInputNode(all_ng_nodes, node_proto);

                    Node node(node_proto);
                    auto axis = node.get_attribute_value<int64_t>("axis");

                    auto concat_op = std::make_shared<op::Concat>(axis);
                    concat_op->set_name(node_proto.output(0));
                    auto concat_gnode = m_graph->add_node_and_edge(concat_op, input_gnodes);
                    NamedNodeVector ret{{node_proto.output(0), concat_gnode}};
                    return ret;
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion
