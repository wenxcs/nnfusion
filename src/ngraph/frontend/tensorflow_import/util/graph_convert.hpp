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

                NamedNodeVector convert_node(const tensorflow::NodeDef& node);

                std::vector<std::shared_ptr<ngraph::Function>> get_funcs();

                std::shared_ptr<nnfusion::graph::Graph> get_graph() { return m_ngraph; }
            private:
                void generate_topology();

                const tensorflow::GraphDef* tf_graph_proto;

                ngraph::op::ParameterVector m_parameters;
                NodeVector m_outputs;
                std::vector<std::shared_ptr<nnfusion::graph::GNode>> m_graph_outputs;
                NodeMap m_ng_node;

                std::shared_ptr<nnfusion::graph::Graph> m_ngraph;
                std::unordered_map<std::shared_ptr<ngraph::Node>,
                                   std::shared_ptr<nnfusion::graph::GNode>>
                    node2gnode_map;

                // node process topology
                std::queue<uint32_t> tf_topology_;
                // pending input count of each node
                std::vector<uint32_t> tf_pending_counts_;
                // the output nodes of each node
                std::vector<std::vector<uint32_t>> tf_node_outputs_;
                // the output node name set
                std::set<std::string> tf_output_name_;
            };
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace ngraph
