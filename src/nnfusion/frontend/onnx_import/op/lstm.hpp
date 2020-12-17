//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "../core/node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateLstmOp(const onnx::NodeProto& node_proto,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    // now only support 7 inputs [X, W, R, B, seq_len, init_h, init_c]
                    GNodeIndexVector input_indexes;
                    for (int i = 0; i < node_proto.input_size(); i++)
                    {
                        if (i != 4)
                        {
                            auto input_desc = GetInputIndex(all_ng_nodes, node_proto, i);
                            input_indexes.push_back(input_desc);
                        }
                    }
                    Node node(node_proto);
                    nnfusion::op::OpConfig::any myConfig;
                    // unsupported attrs: activation related, clip
                    myConfig["direction"] =
                        node.get_attribute_value<std::string>("direction", "forward");
                    myConfig["hidden_size"] = node.get_attribute_value<int64_t>("hidden_size", 0);
                    myConfig["input_forget"] = node.get_attribute_value<int64_t>("input_forget", 0);

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "Lstm", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_indexes, 1);

                    return {{node_proto.output(0), generic_gnode, 0}};
                }

            } // namespace set_1
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace ngraph
