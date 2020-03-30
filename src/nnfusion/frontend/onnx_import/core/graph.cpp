//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <set>

#include "core/tensor.hpp"
#include "graph.hpp"
//#include "node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            Graph::Graph(const onnx::GraphProto& graph_proto)
                : m_graph_proto{&graph_proto}
            {
                /*
                for (const auto& tensor : m_graph_proto->initializer())
                {
                    if (tensor.has_name())
                    {
                        m_initializers.emplace(tensor.name(), Tensor{tensor});
                    }
                }

                // Process all ONNX graph inputs, convert them to nGraph nodes and store in cache
                for (const auto& input : m_graph_proto->input())
                {
                    m_inputs.emplace_back(input);
                    m_ng_node_cache[input.name()] =
                        m_inputs.back().get_ng_node(m_parameters, m_initializers);
                }

                for (const auto& output : m_graph_proto->output())
                {
                    m_outputs.emplace_back(output);
                }

                // Verify that ONNX graph contains only nodes of available operator types
                std::set<std::string> unknown_operator_types;
                for (const auto& node_proto : m_graph_proto->node())
                {
                    if (!m_model->is_operator_available(node_proto))
                    {
                        unknown_operator_types.emplace(detail::to_string(node_proto));
                    }
                }

                NGRAPH_ASSERT(unknown_operator_types.empty())
                    << "unknown operations: " << detail::to_string(unknown_operator_types);

                // Process ONNX graph nodes, convert to nGraph nodes
                for (const auto& node_proto : m_graph_proto->node())
                {
                    m_nodes.emplace_back(node_proto, *this);
                    const Node& node{m_nodes.back()};
                    NodeVector ng_nodes{node.get_ng_nodes()};
                    for (int i = 0; i < ng_nodes.size(); i++)
                    {
                        m_ng_node_cache[node.output(i)] = ng_nodes[i];
                    }
                }*/
            }

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion