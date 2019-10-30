//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <memory>
#include <sstream>
#include <typeindex>
#include <typeinfo>

#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"

using namespace std;
using namespace nnfusion::graph;

atomic<size_t> GNode::m_next_instance_id(0);

GNode::GNode()
    : m_id(-1)
    , m_instance_id(m_next_instance_id.fetch_add(1))
    , m_unique_name("graph_node_" + to_string(m_instance_id))
{
}

GNode::GNode(const std::shared_ptr<ngraph::Node> op_ptr)
    : GNode()
{
    initialize(op_ptr);
}

void GNode::construct_from_op_ptr(const std::shared_ptr<ngraph::Node>& op_ptr)
{
    m_op_ptr = op_ptr;
    m_op_type = op_ptr->description();
    m_name = op_ptr->get_friendly_name();

    m_inputs.clear();
    for (ngraph::descriptor::Input& op_input : op_ptr->get_inputs())
    {
        std::shared_ptr<Input> input =
            std::make_shared<Input>(op_input.get_element_type(), op_input.get_shape());
        m_inputs.push_back(input);
    }
    m_outputs.clear();
    for (ngraph::descriptor::Output& op_output : op_ptr->get_outputs())
    {
        std::shared_ptr<Output> output = std::make_shared<Output>(op_output.get_tensor_ptr());
        m_outputs.push_back(output);
    }
}

void GNode::initialize(const std::shared_ptr<ngraph::Node> op_ptr)
{
    construct_from_op_ptr(op_ptr);
    m_in_edges.clear();
    m_out_edges.clear();
}

const std::string& GNode::get_name() const
{
    if (m_name.empty())
    {
        return m_unique_name;
    }
    return m_name;
}

const std::string& GNode::get_unique_name() const
{
    return m_unique_name;
}

void GNode::set_name(const string& name)
{
    CHECK(m_name.empty()) << "Node name may be set exactly once";
    m_name = name;
}

GNode::~GNode()
{
}

void GNode::add_in_edge(std::shared_ptr<Edge> edge)
{
    m_in_edges.insert(edge);
}

void GNode::add_out_edge(std::shared_ptr<nnfusion::graph::Edge> edge)
{
    m_out_edges.insert(edge);
}

std::vector<std::shared_ptr<nnfusion::graph::Edge>> GNode::get_output_users(size_t i)
{
    std::vector<std::shared_ptr<nnfusion::graph::Edge>> output_users;
    CHECK(i < m_outputs.size()) << "Output index " << i << " is out of range. GNode only has "
                                << m_outputs.size() << " outputs.";
    auto edges = this->get_out_edges();
    for (auto edge : edges)
    {
        if (edge->get_src_output() == i)
        {
            output_users.push_back(edge);
        }
    }
    return output_users;
}

void GNode::reset_op_ptr(const std::shared_ptr<ngraph::Node>& op_ptr)
{
    construct_from_op_ptr(op_ptr);

    // TODO: to be removed once gnode input is used and node input deprecated
    auto edges = this->get_out_edges();
    for (auto& edge : edges)
    {
        CHECK(edge->get_src() == shared_from_this());
        if (edge->is_control_edge())
            continue;
        std::vector<std::shared_ptr<nnfusion::graph::Edge>> ordered_edges;
        for (auto& edge_2 : edge->get_dst()->get_in_edges())
        {
            ordered_edges.push_back(edge_2);
        }
        std::sort(ordered_edges.begin(),
                  ordered_edges.end(),
                  [](const std::shared_ptr<nnfusion::graph::Edge>& a,
                     const std::shared_ptr<nnfusion::graph::Edge>& b) {
                      return (size_t)a->get_dst_input() <
                             (size_t)b->get_dst_input(); // put -1 to the end
                  });

        std::deque<descriptor::Input> m_inputs;
        size_t i = 0;
        for (auto& argument : ordered_edges)
        {
            if (argument->is_control_edge())
                continue;
            for (descriptor::Output& output : argument->get_src()->get_op_ptr()->get_outputs())
            {
                m_inputs.emplace_back(edge->get_dst()->get_op_ptr().get(), i++, output);
            }
        }
        edge->get_dst()->get_op_ptr()->get_inputs() = std::move(m_inputs);
    }
    // end TODO
}

void GNode::Clear()
{
    m_in_edges.clear();
    m_out_edges.clear();
    m_inputs.clear();
    m_outputs.clear();
    m_op_ptr = nullptr;
    m_id = -1;
    m_name.clear();
    m_op_type.clear();
}
