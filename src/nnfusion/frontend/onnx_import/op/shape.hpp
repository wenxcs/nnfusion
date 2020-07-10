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
                NamedNodeVector TranslateShapeOp(const onnx::NodeProto& node_proto,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto data = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto data_shape = data.get_shape();
                    auto op = std::make_shared<op::Constant>(
                        nnfusion::element::i64, Shape{data_shape.size()}, data_shape);
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, nnfusion::graph::GNodeVector{});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace nnfusion
