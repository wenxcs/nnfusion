//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "graph.hpp"
#include "../ops/const.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/exp.hpp"
namespace ngraph
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            NamedNodeVector TranslateIdentityOp(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes)
            {
                NamedNodeVector ret{{node.name(), all_ng_nodes.at(node.input(0))}};
                return ret;
            }

            template <typename T>
            NamedNodeVector TranslateUnaryOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes)
            {
                auto ng_node = std::make_shared<T>(all_ng_nodes.at(node.input(0)));
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            TensorflowGraph::TensorflowGraph(const tensorflow::GraphDef& proto)
                : m_graph_proto{&proto}
            {
                m_map.clear();
                m_map["Const"] =
                    std::bind(TranslateConstOp, std::placeholders::_1, std::placeholders::_2);
                m_map["Identity"] =
                    std::bind(TranslateIdentityOp, std::placeholders::_1, std::placeholders::_2);
                m_map["Abs"] = std::bind(TranslateUnaryOp<ngraph::op::Abs>,
                                         std::placeholders::_1,
                                         std::placeholders::_2);

                m_map["Exp"] = std::bind(TranslateUnaryOp<ngraph::op::Exp>,
                                         std::placeholders::_1,
                                         std::placeholders::_2);

                std::cerr << "Converting Tensorflow Graph" << std::endl;

                generate_topology();
                for (const auto& node_proto : proto.node())
                {
                    auto ng_nodes = convert_node(node_proto);
                    for (auto& node : ng_nodes)
                    {
                        m_ng_node[node.first] = node.second;
                    }
                    if (is_input.find(node_proto.name()) != is_input.end())
                    {
                        m_inputs.emplace_back(ng_nodes.front().second);
                    }
                    if (is_output.find(node_proto.name()) != is_output.end())
                    {
                        m_outputs.emplace_back(ng_nodes.back().second);
                    }
                }
            }

            void TensorflowGraph::generate_topology()
            {
                for (const auto& node_proto : m_graph_proto->node())
                    out_edges_count[node_proto.name()] = 0;

                for (const auto& node_proto : m_graph_proto->node())
                {
                    in_edges_count[node_proto.name()] = node_proto.input_size();
                    for (auto& input : node_proto.input())
                    {
                        ++out_edges_count[input];
                    }
                }

                for (auto& it : in_edges_count)
                    if (it.second == 0)
                        is_input.insert(it.first);

                for (auto& it : out_edges_count)
                    if (it.second == 0)
                        is_output.insert(it.first);
            }

            NamedNodeVector TensorflowGraph::convert_node(const tensorflow::NodeDef& node)
            {
                auto func = m_map.find(node.op());
                if (func != m_map.end())
                {
                    return func->second(node, m_ng_node);
                }
                else
                {
                    std::cerr << "Unsupport operator: " << node.op() << std::endl;
                    return NamedNodeVector{};
                }
            }

            std::shared_ptr<ngraph::Function> TensorflowGraph::get_outputs()
            {
                auto ng_function =
                    std::make_shared<ngraph::Function>(m_outputs, ngraph::op::ParameterVector{});
                return ng_function;
            }
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace ngraph
