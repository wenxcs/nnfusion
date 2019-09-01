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

#include "nnfusion/core/graph/edge.hpp"
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

void GNode::initialize(const std::shared_ptr<ngraph::Node> op_ptr)
{
    m_op_ptr = op_ptr;
    m_op_type = op_ptr->description();
    m_name = op_ptr->get_name();

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
    if (m_name.empty())
    {
        m_name = name;
    }
    else
    {
        throw ngraph::ngraph_error("Node name may be set exactly once");
    }
}

GNode::~GNode()
{
}

const std::set<std::shared_ptr<nnfusion::graph::Edge>>& GNode::get_in_edges() const
{
    return m_in_edges;
}

void GNode::add_in_edge(std::shared_ptr<Edge> edge)
{
    m_in_edges.insert(edge);
}

const std::set<std::shared_ptr<nnfusion::graph::Edge>>& GNode::get_out_edges() const
{
    return m_out_edges;
}

void GNode::add_out_edge(std::shared_ptr<nnfusion::graph::Edge> edge)
{
    m_out_edges.insert(edge);
}

void GNode::reset_op_ptr(const std::shared_ptr<ngraph::Node>& node)
{
    // Todo:
    //	1) update node->get_outputs();
    // 	2) handle upstream/downstream nodes (includes control dependencies);

    this->m_op_ptr = node;
    this->m_op_type = node->description();
    this->m_name = node->get_name();

    auto edges = this->get_out_edges();
    for (auto& edge : edges)
    {
        enforce(edge->get_src() == shared_from_this());
        if (edge->is_control_edge())
            continue;
        size_t i = 0;
        std::deque<descriptor::Input> m_inputs;
        for (auto& argument : edge->get_dst()->get_in_edges())
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
}

void GNode::Clear()
{
    m_in_edges.clear();
    m_out_edges.clear();
    m_op_ptr = nullptr;
    m_id = -1;
    m_name.clear();
    m_op_type.clear();
}
