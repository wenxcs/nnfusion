//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "nnfusion/core/operators/op_define/add.hpp"

#include "core/node.hpp"
#include "utils/broadcasting.hpp"

namespace nnfusion
{
    namespace frontend namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline add(const onnx::NodeProto& node,
                           const NodeMap& all_ng_nodes,
                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto axis = node.get_attribute_value<int64_t>("axis", 0);
                    NodeVector ng_inputs{legacy_style_broadcast_for_binary_operation(
                        node.get_ng_inputs().at(0), node.get_ng_inputs().at(1), axis)};

                    return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
                }

            } // namespace set_1

            namespace set_7
            {
                inline NodeVector add(const Node& node)
                {
                    NodeVector ng_inputs{
                        numpy_style_broadcast_for_binary_operation(node.get_ng_inputs())};
                    return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import
} // namespace frontend
} // namespace nnfusion
