//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------
#include "transpose.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector
                    TranslateTransposeOp(const onnx::NodeProto& node_proto,
                                         const NodeMap& all_ng_nodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto data = GetInputIndex(all_ng_nodes, node_proto, 0);

                    auto input_rank = data.get_shape().size();
                    Node node(node_proto);
                    auto perm = node.get_attribute_value<std::vector<int64_t>>("perm");
                    if (perm.empty())
                    {
                        perm.resize(input_rank);
                        // by default it reverse input dims
                        std::iota(perm.rbegin(), perm.rend(), 0);
                    }
                    AxisVector ng_axis_order(perm.begin(), perm.end());

                    auto out_gnode =
                        nnfusion::graph::numpy_transpose(data.gnode, ng_axis_order, data.index);
                    out_gnode->get_op_ptr()->set_name(node_proto.output(0));
                    out_gnode->set_name(node_proto.output(0));
                    m_graph->add_node(out_gnode);
                    m_graph->add_edge(data.gnode, data.index, out_gnode, 0);

                    return {{node_proto.output(0), out_gnode}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
