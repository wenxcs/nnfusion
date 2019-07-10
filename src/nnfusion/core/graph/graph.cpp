// Microsoft (c) 2019, NNFusion Team

#include <sstream>

#include "graph.hpp"

using namespace nnfusion::graph;

// Graph

Graph::Graph()
{
    // TODO: need add source to sink control edge??
}

Graph::~Graph()
{
    // TODO: release node
}

void Graph::add_node(std::shared_ptr<Node> node)
{
    const size_t id = m_nodes.size();
    node->set_id(id);
    m_nodes.push_back(node);
    ++m_num_nodes;
}

std::shared_ptr<Node> Graph::copy_node(const std::shared_ptr<Node> node)
{
    std::shared_ptr<Node> copy = node->copy_with_new_args(node->get_arguments());

    // todo: how to copy a node????

    return copy;
}

void Graph::remove_node(std::shared_ptr<Node> node)
{
    //TF_DCHECK_OK(IsValidNode(node)) << node->DebugString();
    //DCHECK(!node->IsSource());
    //DCHECK(!node->IsSink());

    // Remove any edges involving this node.
    while (!node->get_in_edges().empty())
    {
        remove_edge(*node->get_in_edges().begin());
    }
    while (!node->get_out_edges().empty())
    {
        remove_edge(*node->get_out_edges().begin());
    }
    m_nodes[node->get_id()] = nullptr;
    --m_num_nodes;
    node->Clear();
}

const std::shared_ptr<Edge>
    Graph::add_edge(std::shared_ptr<Node> source, int x, std::shared_ptr<Node> dest, int y)
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

void Graph::remove_edge(std::shared_ptr<Edge> e)
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

void Graph::set_default_output_nodes()
{
    m_output_nodes.clear();
    for (auto node : m_nodes)
    {
        if (node != nullptr && node->get_output_size() == 0)
        {
            m_output_nodes.push_back(node);
        }
    }
}
