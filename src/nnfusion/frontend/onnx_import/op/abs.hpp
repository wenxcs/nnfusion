//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <memory>

#include "nnfusion/core/operators/op_define/abs.hpp"

#include "../core/node.hpp"
#include "../util/util.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace op
            {
                namespace set_1
                {
                    inline NamedNodeVector abs(const onnx::NodeProto& node,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
                    {
                        auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                        auto ng_node = std::make_shared<::op::Abs>();
                        NNFUSION_CHECK(node.output_size() == 1)
                            << "Abs should only has one output.";
                        ng_node->set_name(node.output(0));
                        auto gnode = m_graph->add_node_and_edge(ng_node, {input_gnode});
                        NamedNodeVector ret{{node.output(0), gnode}};
                        return ret;
                    }

                } // namespace set_1

            } //namespace op

        } // namespace onnx_import
    }     // namespace frontedn
} // namespace ngraph
