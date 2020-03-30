//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "graph_convert.hpp"
#include "ops_bridge.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            GraphConvert::GraphConvert(const onnx::ModelProto& proto)
                : onnx_model_proto{&proto}
                , onnx_graph_proto(&(proto.graph()))
                , m_graph(new nnfusion::graph::Graph())
            {
                // Note: onnx connect nodes by tensor's name instead of op name
                /*
                ir_version: 3
                producer_name: "ngraph ONNXImporter"
                graph {
                node {
                    input: "A"
                    input: "B"
                    output: "X"
                    name: "add_node1"
                    op_type: "Add"
                }
                node {
                    input: "X"
                    input: "C"
                    output: "Y"
                    name: "add_node2"
                    op_type: "Add"
                }
                name: "test_graph"
                input {
                    name: "A"
                    type {
                    tensor_type {
                        elem_type: FLOAT
                        shape {
                        dim {
                            dim_value: 1
                        }
                        }
                    }
                    }
                }
                input {
                    name: "B"
                    type {
                    tensor_type {
                        elem_type: FLOAT
                        shape {
                        dim {
                            dim_value: 1
                        }
                        }
                    }
                    }
                }
                input {
                    name: "C"
                    type {
                    tensor_type {
                        elem_type: FLOAT
                        shape {
                        dim {
                            dim_value: 1
                        }
                        }
                    }
                    }
                }
                output {
                    name: "Y"
                    type {
                    tensor_type {
                        elem_type: FLOAT
                        shape {
                        dim {
                            dim_value: 1
                        }
                        }
                    }
                    }
                }
                }
                opset_import {
                version: 4
                }
                */
                NNFUSION_LOG(INFO) << "Converting Onnx Graph";
                // Walk through the elements of opset_import field and register operator sets
                // for each domain. An exception UnknownDomain() will raise if the domain is
                // unknown or invalid.
                for (const auto& id : onnx_model_proto->opset_import())
                {
                    m_domain_convert_func_map.emplace(
                        id.domain(),
                        OperatorsBridge::get_convert_func_map(
                            id.version(), (id.domain() == "ai.onnx" ? "" : id.domain())));
                }
                // onnx.proto(.3): the empty string ("") for domain or absence of opset_import field
                // implies the operator set that is defined as part of the ONNX specification.
                const auto dm = m_domain_convert_func_map.find("");
                if (dm == std::end(m_domain_convert_func_map))
                {
                    m_domain_convert_func_map.emplace(
                        "", OperatorsBridge::get_convert_func_map(ONNX_OPSET_VERSION, ""));
                }

                m_graph = std::make_shared<nnfusion::graph::Graph>();

                for (const auto& tensor : onnx_graph_proto->initializer())
                {
                    if (tensor.has_name())
                    {
                        m_initializers.emplace(tensor.name(), Tensor{tensor});
                    }
                }

                // Process all ONNX graph inputs, convert them to NNFusion nodes
                for (const auto& input_proto : onnx_graph_proto->input())
                {
                    ValueInfo input_value_info(input_proto);
                    const auto it = m_initializers.find(input_proto.name());
                    std::shared_ptr<graph::GNode> input_gnode;
                    if (it != std::end(m_initializers))
                    {
                        auto input_op =
                            make_constant_op(input_proto.type().tensor_type().elem_type(),
                                             input_value_info.get_shape(),
                                             it->second);
                        input_op->set_name(input_proto.name());
                        input_gnode = m_graph->add_node_and_edge(input_op, graph::GNodeVector({}));
                    }
                    else
                    {
                        auto input_op = std::make_shared<op::Parameter>(
                            input_value_info.get_element_type(), input_value_info.get_shape());
                        input_op->set_name(input_proto.name());
                        input_gnode = m_graph->add_node_and_edge(input_op, graph::GNodeVector({}));
                    }

                    m_node_map[input_proto.name()] = {input_gnode};
                }

                for (const auto& output : onnx_graph_proto->output())
                {
                    m_output_names.insert(output.name());
                }

                // Verify that ONNX graph contains only nodes of available operator types
                for (const auto& node_proto : onnx_graph_proto->node())
                {
                    NNFUSION_CHECK(is_operator_available(node_proto)) << "unknown operations: "
                                                                      << node_proto.DebugString();
                }

                // Process ONNX graph nodes, convert to nGraph nodes
                for (const auto& node_proto : onnx_graph_proto->node())
                {
                    auto results = convert_node(node_proto);
                    for (auto& name_gnode_pair : results)
                    {
                        m_node_map[name_gnode_pair.first] = {name_gnode_pair.second};

                        if (m_output_names.find(name_gnode_pair.first) != m_output_names.end())
                        {
                            m_graph_outputs.emplace_back(name_gnode_pair.second);
                        }
                    }
                }

                m_graph->set_default_parameters();
                m_graph->set_outputs(m_graph_outputs);

                NNFUSION_LOG(INFO) << "convert graph done";
            }

            NamedNodeVector GraphConvert::convert_node(const onnx::NodeProto& node)
            {
                NamedNodeVector ret =
                    get_convert_func(node.op_type(), node.domain())(node, m_node_map, m_graph);

                return std::move(ret);
            }

            const ConvertFunc& GraphConvert::get_convert_func(const std::string& name,
                                                              const std::string& domain) const
            {
                const auto dm = m_domain_convert_func_map.find(domain);
                NNFUSION_CHECK(dm != std::end(m_domain_convert_func_map)) << "Unknown Domain: "
                                                                          << domain;

                const auto op = dm->second.find(name);
                NNFUSION_CHECK(op != std::end(dm->second))
                    << "Unknown ConvertFunc: " << (domain.empty() ? "" : domain + ".") << name;

                return op->second;
            }

            bool GraphConvert::is_operator_available(const onnx::NodeProto& node_proto) const
            {
                const auto dm = m_domain_convert_func_map.find(node_proto.domain());

                if (dm == std::end(m_domain_convert_func_map))
                {
                    return false;
                }
                const auto op = dm->second.find(node_proto.op_type());
                return (op != std::end(dm->second));
            }

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
