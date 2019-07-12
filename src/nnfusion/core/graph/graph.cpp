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

void Graph::add_node(std::shared_ptr<GNode> node)
{
    const size_t id = m_nodes.size();
    node->set_id(id);
    m_nodes.push_back(node);
    ++m_node_size;
}

std::shared_ptr<GNode> Graph::add_node(const std::shared_ptr<ngraph::Node> node)
{
    std::shared_ptr<GNode> graph_node;
    if (m_free_nodes.empty())
    {
        graph_node = std::make_shared<GNode>();
    }
    else
    {
        graph_node = m_free_nodes.back();
        m_free_nodes.pop_back();
    }
    graph_node->initialize(node);
    add_node(graph_node);
    return graph_node;
}

std::shared_ptr<GNode> Graph::copy_node(const std::shared_ptr<GNode> node)
{
    std::shared_ptr<GNode> copy = add_node(node->get_op_ptr());

    // todo: how about edge???

    return copy;
}

void Graph::remove_node(std::shared_ptr<GNode> node)
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
    node->Clear();
    m_free_nodes.push_back(node);
    --m_node_size;
}

const std::shared_ptr<Edge>
    Graph::add_edge(std::shared_ptr<GNode> source, int x, std::shared_ptr<GNode> dest, int y)
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

    std::shared_ptr<Edge> edge;

    if (m_free_edges.empty())
    {
        edge = std::make_shared<Edge>(); // placement new
    }
    else
    {
        edge = m_free_edges.back();
        m_free_edges.pop_back();
    }
    edge->m_id = m_edges.size();
    edge->m_src = source;
    edge->m_dst = dest;
    edge->m_src_output = x;
    edge->m_dst_input = y;
    source->add_out_edge(edge);
    dest->add_in_edge(edge);
    m_edges.push_back(edge);

    ++m_edge_size;
    return edge;
}

void Graph::remove_edge(std::shared_ptr<Edge> edge)
{
    //TF_DCHECK_OK(IsValidNode(e->src_)) << e->src_->DebugString();
    //TF_DCHECK_OK(IsValidNode(e->dst_)) << e->dst_->DebugString();
    edge->get_src()->remove_out_edge(edge);
    edge->get_dst()->remove_in_edge(edge);
    //CHECK_EQ(e, m_edges[e->m_id]);
    //CHECK_GT(m_num_edges, 0);

    m_edges[edge->m_id] = nullptr;

    edge->m_src = nullptr;
    edge->m_dst = nullptr;
    edge->m_id = -1;
    edge->m_src_output = kControlSlot - 1;
    edge->m_dst_input = kControlSlot - 1;
    m_free_edges.push_back(edge);
    --m_edge_size;
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
