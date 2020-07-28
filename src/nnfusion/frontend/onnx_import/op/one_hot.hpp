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
                NamedNodeVector TranslateOneHotOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    GNodeIndexVector input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto indices = input_indexes[0];
                    auto depth = input_indexes[1];
                    auto off_on_values = input_indexes[2];

                    std::vector<int64> depth_vec;
                    NNFUSION_CHECK(GetValueFromNGraphOp(depth.gnode, &depth_vec));
                    NNFUSION_CHECK(depth_vec.size() == 1);

                    std::vector<int64> off_on_values_vec;
                    NNFUSION_CHECK(GetValueFromNGraphOp(off_on_values.gnode, &off_on_values_vec));
                    NNFUSION_CHECK(off_on_values_vec.size() == 2);

                    Node node(node_proto);
                    auto axis = node.get_attribute_value<int64>("axis", -1);

                    auto type_str = nnfusion::element::i64.c_type_string();

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["axis"] = axis;
                    myConfig["depth"] = depth_vec[0];
                    myConfig["off_value"] = off_on_values_vec[0];
                    myConfig["on_value"] = off_on_values_vec[1];
                    myConfig["T"] = type_str;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0), "OneHot", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, {indices});

                    return {{node_proto.output(0), generic_gnode}};
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion
