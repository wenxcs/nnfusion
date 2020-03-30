//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

#include "../onnx_base.hpp"

#include "value_info.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            class Graph
            {
            public:
                Graph(const onnx::GraphProto& proto);

                //const std::vector<Node>& get_nodes() const { return m_nodes; }
                //const std::vector<ValueInfo>& get_inputs() const { return m_inputs; }
                //const std::vector<ValueInfo>& get_outputs() const { return m_outputs; }

                const std::string& get_name() const { return m_graph_proto->name(); }
                //NodeVector make_ng_nodes(const Node& node) const
                //{
                //    return m_model->get_operator(node.op_type(), node.domain())(node);
                //}

            private:
                const onnx::GraphProto* m_graph_proto;
                //std::vector<Node> m_nodes;
                //std::vector<ValueInfo> m_inputs;
                //std::vector<ValueInfo> m_outputs;
                //std::map<std::string, Tensor> m_initializers;
            };

            inline std::ostream& operator<<(std::ostream& outs, const Graph& graph)
            {
                return (outs << "<Graph: " << graph.get_name() << ">");
            }

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
