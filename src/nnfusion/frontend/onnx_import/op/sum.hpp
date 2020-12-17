//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "core/node.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateSumOp(const onnx::NodeProto& node_proto,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    nnfusion::op::OpConfig::any myConfig;

                    // Since Ngraph doesn't have AddN, so we use GenericOp to
                    // represent the AddN.
                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0), "AddN", myConfig);

                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_indexes);
                    // Return the node vecoter, this is one tf-node to one nnfusion-node case,
                    // if your code converts one tf-node into several nnfusion-nodes, you can
                    // refer BiasAdd, which is converted to Broadcast and Add;
                    NamedNodeVector ret{{node_proto.output(0), generic_gnode}};
                    return ret;
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace nnfusion
