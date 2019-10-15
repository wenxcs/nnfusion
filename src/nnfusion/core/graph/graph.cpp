// Microsoft (c) 2019, NNFusion Team

#include <sstream>

#include "graph.hpp"
#include "graph_util.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion::graph;

std::atomic<size_t> Graph::m_next_instance_id(0);

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

Graph::Graph(const std::string& name)
    : m_instance_id(m_next_instance_id.fetch_add(1))
    , m_temporary_pool_size(0)
    , m_name(name)
    , m_unique_name("Graph_" + std::to_string(m_instance_id))
{
    // TODO: need add source to sink control edge??
}

Graph::Graph(const std::shared_ptr<ngraph::Function>& func, const std::string& name)
    : m_instance_id(m_next_instance_id.fetch_add(1))
    , m_temporary_pool_size(0)
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

const std::string& Graph::get_friendly_name() const
{
    if (m_name.empty())
    {
        return m_unique_name;
    }
    return m_name;
}

const std::string& Graph::get_name() const
{
    return m_unique_name;
}

void Graph::set_name(const std::string& name)
{
    CHECK(m_name.empty()) << "Graph name may be set exactly once.";

    m_name = name;
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

std::vector<std::shared_ptr<GNode>> Graph::get_ordered_ops(bool include_control_deps)
{
    // todo: stored ops instead of calculate each time
    std::vector<std::shared_ptr<GNode>> nodes;
    ReverseDFS(this,
               get_outputs(),
               nullptr,
               [&](std::shared_ptr<GNode> node) { nodes.push_back(node); },
               NodeComparatorName());
    std::vector<std::shared_ptr<GNode>> update_nodes;
    for (auto node : nodes)
    {
        if (node->get_op_type() == "Constant")
        {
            update_nodes.push_back(node);
        }
    }
    for (auto node : nodes)
    {
        if (node->get_op_type() != "Constant")
        {
            update_nodes.push_back(node);
        }
    }
    return update_nodes;
}

std::vector<std::shared_ptr<GNode>> Graph::get_const_nodes()
{
    std::vector<std::shared_ptr<GNode>> const_nodes;
    for (auto node : get_nodes())
    {
        if (node->get_op_type() == "Constant")
        {
            const_nodes.push_back(node);
        }
    }
    return const_nodes;
}

const std::shared_ptr<nnfusion::graph::Edge>
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

    // control slots must only be linked to control slots
    if (x == kControlSlot || y == kControlSlot)
    {
        CHECK(x == kControlSlot);
        CHECK(y == kControlSlot);
    }

    std::shared_ptr<Edge> edge = nullptr;

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

bool Graph::find_edge(std::shared_ptr<GNode> source, int x, std::shared_ptr<GNode> dest, int y)
{
    for (const auto edge : m_edges)
    {
        if (edge->get_src() == source && edge->get_dst() == dest && edge->get_src_output() == x &&
            edge->get_dst_input() == y)
        {
            return true;
        }
    }
    return false;
}

const std::shared_ptr<nnfusion::graph::Edge> Graph::add_control_edge(std::shared_ptr<GNode> source,
                                                                     std::shared_ptr<GNode> dest,
                                                                     bool allow_duplicates)
{
    if (!allow_duplicates)
    {
        for (const auto edge : dest->get_in_edges())
        {
            if (edge->is_control_edge() && edge->get_src() == source)
            {
                // the requested edge already exist
                return nullptr;
            }
        }
    }

    return add_edge(source, kControlSlot, dest, kControlSlot);
}

void Graph::remove_edge(std::shared_ptr<Edge> edge)
{
    //TF_DCHECK_OK(IsValidNode(e->src_)) << e->src_->DebugString();
    //TF_DCHECK_OK(IsValidNode(e->dst_)) << e->dst_->DebugString();
    edge->get_src()->remove_out_edge(edge);
    edge->get_dst()->remove_in_edge(edge);
    CHECK(edge == m_edges[edge->m_id]);
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

void Graph::remove_control_edge(std::shared_ptr<Edge> edge)
{
    // todo remove ^src from dst's node_def's input
    remove_edge(edge);
}

void Graph::set_default_outputs()
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
void Graph::set_outputs(std::vector<std::shared_ptr<GNode>> outputs)
{
    m_output_nodes = outputs;
}

std::vector<std::shared_ptr<GNode>> Graph::get_outputs()
{
    return m_output_nodes;
}

const size_t Graph::get_output_size()
{
    return m_output_nodes.size();
}

const std::shared_ptr<GNode> Graph::get_output_op(size_t i)
{
    return m_output_nodes.at(i);
}

void Graph::set_default_parameters()
{
    m_parameters.clear();
    for (auto node : m_nodes)
    {
        if (node != nullptr && node->get_op_ptr()->is_parameter())
        {
            m_parameters.push_back(node);
        }
    }
}

std::vector<std::shared_ptr<GNode>> Graph::get_parameters()
{
    if (m_parameters.empty())
    {
        set_default_parameters();
    }
    return m_parameters;
}

size_t Graph::get_temporary_pool_size()
{
    return m_temporary_pool_size;
}

void Graph::set_temporary_pool_size(size_t size)
{
    m_temporary_pool_size = size;
}
