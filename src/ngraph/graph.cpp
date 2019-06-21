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

#include <sstream>

#include "ngraph/graph.hpp"
#include "ngraph/node.hpp"

using namespace ngraph;

const int Graph::kControlSlot = -1;

std::string Edge::DebugString() const
{
    std::stringstream ss;
    ss << "[id=" << m_id << " " << m_src->get_name().c_str() << ":" << m_src_output << " -> "
       << m_dst->get_name().c_str() << ":" << m_dst_input << "]";
    return ss.str();
}

// Graph

Graph::Graph()
{
    // TODO: need add source to sink control edge??
}

Graph::~Graph()
{
    // TODO: release node
}

void Graph::AddNode(std::shared_ptr<Node> node)
{
    const size_t id = m_nodes.size();
    node->set_id(id);
    m_nodes.push_back(node);
    ++m_num_nodes;
}

void Graph::RemoveNode(std::shared_ptr<Node> node)
{
    //TF_DCHECK_OK(IsValidNode(node)) << node->DebugString();
    //DCHECK(!node->IsSource());
    //DCHECK(!node->IsSink());

    // Remove any edges involving this node.
    while (!node->get_in_edges().empty())
    {
        RemoveEdge(*node->get_in_edges().begin());
    }
    while (!node->get_out_edges().empty())
    {
        RemoveEdge(*node->get_out_edges().begin());
    }
    m_nodes[node->get_id()] = nullptr;
    --m_num_nodes;
    node->Clear();
}

const std::shared_ptr<Edge>
    Graph::AddEdge(std::shared_ptr<Node> source, int x, std::shared_ptr<Node> dest, int y)
{
    //TF_DCHECK_OK(IsValidNode(source)) << source->DebugString();
    //TF_DCHECK_OK(IsValidNode(dest)) << dest->DebugString();

    //// source/sink must only be linked via control slots, and
    //// control slots must only be linked to control slots.
    //if (source == source_node() || dest == sink_node() || x == kControlSlot ||
    //y == kControlSlot) {
    //DCHECK_EQ(x, kControlSlot) << source->DebugString();
    //DCHECK_EQ(y, kControlSlot) << dest->DebugString();
    //}

    std::shared_ptr<Edge> e;

    if (m_free_edges.empty())
    {
        e = std::make_shared<Edge>(); // placement new
    }
    else
    {
        e = m_free_edges.back();
        m_free_edges.pop_back();
    }
    e->m_id = m_edges.size();
    e->m_src = source;
    e->m_dst = dest;
    e->m_src_output = x;
    e->m_dst_input = y;
    source->add_out_edge(e);
    dest->add_in_edge(e);
    m_edges.push_back(e);

    ++m_num_edges;
    return e;
}

void Graph::RemoveEdge(std::shared_ptr<Edge> e)
{
    //TF_DCHECK_OK(IsValidNode(e->src_)) << e->src_->DebugString();
    //TF_DCHECK_OK(IsValidNode(e->dst_)) << e->dst_->DebugString();
    e->get_src()->remove_out_edge(e);
    e->get_dst()->remove_in_edge(e);
    //CHECK_EQ(e, m_edges[e->m_id]);
    //CHECK_GT(m_num_edges, 0);

    m_edges[e->m_id] = nullptr;

    e->m_src = nullptr;
    e->m_dst = nullptr;
    e->m_id = -1;
    e->m_src_output = kControlSlot - 1;
    e->m_dst_input = kControlSlot - 1;
    m_free_edges.push_back(e);
    --m_num_edges;
}