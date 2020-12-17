//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <vector>

#include "../../util/evaluator.hpp"
#include "../util/util.hpp"
#include "expand.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateExpandOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    GNodeIndexVector input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    auto input = input_indexes[0];

                    auto input_shape = input.get_shape();
                    size_t input_rank = input_shape.size();
                    std::vector<int64> expand_shape;
                    NNFUSION_CHECK(GetValueFromNGraphOp(input_indexes[1].gnode, &expand_shape));

                    auto expand_shape_op = std::make_shared<op::Constant>(
                        element::i32,
                        Shape(expand_shape.begin(), expand_shape.end()),
                        std::vector<int>({1}));
                    auto expand_shape_gnode =
                        m_graph->add_node_and_edge(expand_shape_op, GNodeVector({}));

                    auto expand_shape_index = GNodeIndex{expand_shape_gnode, 0};

                    std::tie(input, expand_shape_index) =
                        numpy_broadcast(std::make_pair(input, expand_shape_index), m_graph);
                    // TODO: might reset name?
                    if (input.gnode != input_indexes[0].gnode)
                    {
                        input.gnode->get_op_ptr()->set_name(node_proto.output(0));
                        input.gnode->set_name(node_proto.output(0));
                    }
                    return {{node_proto.output(0), input}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
