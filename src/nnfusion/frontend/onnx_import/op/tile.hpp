//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "../../util/evaluator.hpp"
#include "core/node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateTileOp(const onnx::NodeProto& node_proto,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    GNodeIndexVector input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    nnfusion::op::OpConfig::any myConfig;
                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0), "Tile", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_indexes);

                    return {{node_proto.output(0), generic_gnode, 0}};
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion
