// Microsoft (c) 2019, NNFusion Team

#include <sstream>

#include "graph.hpp"

using namespace nnfusion::graph;

std::atomic<size_t> Graph::m_next_instance_id(0);

Graph::Graph(const std::string& name)
    : m_instance_id(m_next_instance_id.fetch_add(1))
    , m_name(name)
    , m_unique_name("Graph_" + std::to_string(m_instance_id))
{
    // TODO: need add source to sink control edge??
}

Graph::Graph(const ngraph::op::ParameterVector& parameters,
             const std::vector<std::shared_ptr<GNode>>& outputs,
             const std::string& name)
    : m_instance_id(m_next_instance_id.fetch_add(1))
    , m_name(name)
    , m_unique_name("Graph_" + std::to_string(m_instance_id))
{
    // TODO: need add source to sink control edge??
}

// Graph
void construct_graph_from_io_nodes(Graph* graph,
                                   const std::vector<std::shared_ptr<ngraph::Node>>& io_nodes,
                                   bool include_control_deps)
{
    // Stack of work to do.
    struct Work
    {
        std::shared_ptr<ngraph::Node> node;
        bool leave; // Are we entering or leaving node?
    };

    std::vector<Work> stack(io_nodes.size());

    for (int i = 0; i < io_nodes.size(); ++i)
    {
        stack[i] = Work{io_nodes[i], false};
    }

    std::unordered_set<std::shared_ptr<ngraph::Node>> visited;
    std::unordered_map<std::shared_ptr<ngraph::Node>, std::shared_ptr<GNode>> node_convert;
    while (!stack.empty())
    {
        Work w = stack.back();
        stack.pop_back();

        auto node = w.node;

        if (w.leave)
        {
            // TODO: add edge
            int dst_output = 0;
            for (auto arg : node->get_arguments())
            {
                graph->add_edge(node_convert[arg], 0, node_convert[node], dst_output);
                dst_output++;
            }

            if (include_control_deps)
            {
                for (auto cdep : node->get_control_dependencies())
                {
                    graph->add_edge(node_convert[cdep], -1, node_convert[node], -1);
                }
            }
            continue;
        }

        if (visited.count(node) > 0)
        {
            continue;
        }
        visited.insert(node);
        auto gnode = graph->add_node(node);
        node_convert[node] = gnode;
        stack.push_back(Work{node, true});

        auto add_work = [&visited, &stack](std::shared_ptr<ngraph::Node> in_node) {
            if (visited.count(in_node) == 0)
            {
                // Note; we must not mark as visited until we actually process it.
                stack.push_back(Work{in_node, false});
            }
        };

        for (auto arg : node->get_arguments())
        {
            add_work(arg);
        }

        if (include_control_deps)
        {
            for (auto cdep : node->get_control_dependencies())
            {
                add_work(cdep);
            }
        }
    }
}

Graph::Graph(const std::shared_ptr<ngraph::Function>& func, const std::string& name)
    : m_instance_id(m_next_instance_id.fetch_add(1))
    , m_name(name)
    , m_unique_name("Graph_" + std::to_string(m_instance_id))
{
    std::vector<std::shared_ptr<ngraph::Node>> nodes;

    for (auto r : func->get_results())
    {
        nodes.push_back(r);
    }

    for (auto param : func->get_parameters())
    {
        nodes.push_back(param);
    }

    construct_graph_from_io_nodes(this, nodes, true /*include control dependencies*/);
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

std::vector<std::shared_ptr<GNode>> Graph::get_nodes()
{
    std::vector<std::shared_ptr<GNode>> valid_nodes;
    for (auto node : m_nodes)
    {
        if (node != nullptr)
        {
            valid_nodes.push_back(node);
        }
    }
    return valid_nodes;
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
