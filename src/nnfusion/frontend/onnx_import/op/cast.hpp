//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "../util/util.hpp"
#include "core/node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateCastOp(const onnx::NodeProto& node_proto,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    Node node(node_proto);
                    int64_t target_type = node.get_attribute_value<int64_t>("to");
                    element::Type et_type;
                    ONNXDataTypeToNNFusionElementType(
                        static_cast<onnx::TensorProto_DataType>(target_type), &et_type);

                    auto op = std::make_shared<op::Convert>(et_type);
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, {input_gnode});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion
