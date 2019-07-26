//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <queue>
#include <string>
#include <vector>

#include "../tensorflow_base.hpp"
#include "ngraph/op/parameter_vector.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "util.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            class GraphConvert
            {
            public:
                GraphConvert(const tensorflow::GraphDef& proto);

                // const std::vector<Node>& get_nodes() const { return m_nodes; }
                // const std::vector<ValueInfo>& get_inputs() const { return m_inputs; }
                // const std::vector<ValueInfo>& get_outputs() const { return m_outputs; }
                const ngraph::op::ParameterVector& get_ng_parameters() const
                {
                    return m_parameters;
                }

                std::vector<std::shared_ptr<ngraph::Node>>
                    get_ng_node(const std::string& name) const
                {
                    return m_ng_node.at(name);
                }

                NamedNodeVector convert_node(const tensorflow::NodeDef& node);

                std::vector<std::shared_ptr<ngraph::Function>> get_outputs();

                std::shared_ptr<nnfusion::graph::Graph> get_graph() { return m_ngraph; }
            private:
                void generate_topology();

                const tensorflow::GraphDef* m_graph_proto;
                // std::vector<TensorflowNode> m_nodes;
                // std::vector<ValueInfo> m_inputs;
                // std::vector<ValueInfo> m_outputs;
                ngraph::op::ParameterVector m_parameters;
                std::map<std::string, uint32_t> in_edges_count;
                std::map<std::string, uint32_t> out_edges_count;
                NodeVector m_inputs;
                NodeVector m_outputs;
                std::set<std::string> is_input;
                std::set<std::string> is_output;
                NodeMap m_ng_node;
                std::shared_ptr<nnfusion::graph::Graph> m_ngraph;

                std::queue<uint32_t> topology_;
                std::unordered_map<std::string, uint32_t> tensorflow_name2nodeIdx_map_;
                std::vector<uint32_t> pending_counts_;
                std::vector<std::vector<uint32_t>> tensorflow_node_outputs_;
            };

            // inline std::ostream& operator<<(std::ostream& outs, const Graph& graph)
            // {
            //     return (outs << "<Graph: " << graph.get_name() << ">");
            // }

        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace ngraph
