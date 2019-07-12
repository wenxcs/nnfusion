// Microsoft (c) 2019, NNFusion Team

#pragma once

#include <memory>
#include <vector>
#include "edge.hpp"
#include "gnode.hpp"

namespace nnfusion
{
    namespace graph
    {
        // Thread compatible but not thread safe.
        class Graph
        {
        public:
            // Constructs a graph with a single SOURCE (always id kSourceId) and a
            // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.
            explicit Graph();

            ~Graph();

            static const int kControlSlot = -1;

            // Adds a new node to this graph, and returns it. Infers the Op and
            // input/output types for the node. *this owns the returned instance.
            // Returns nullptr and sets *status on error.
            void add_node(std::shared_ptr<GNode> node);

            std::shared_ptr<GNode> add_node(const std::shared_ptr<ngraph::Node> node);

            // Removes a node from this graph, including all edges from or to it.
            // *node should not be accessed after calling this function.
            // REQUIRES: node->IsOp()
            void remove_node(std::shared_ptr<GNode> node);

            // Copies node, which may belong to another graph, to a new node,
            // which is returned.  Does not copy any edges.  *this owns the
            // returned instance.
            std::shared_ptr<GNode> copy_node(const std::shared_ptr<GNode> node);

            // The number of live nodes in the graph.
            //
            // Because nodes can be removed from the graph, get_node_size() is often
            // smaller than get_max_node_id(). If one needs to create an array of
            // nodes indexed by node ids, get_max_node_id() should be used as the
            // array's size.
            size_t get_node_size() const { return m_node_size; }
            // Returns one more than the maximum id assigned to any node.
            size_t get_max_node_id() const { return m_nodes.size(); }
            // Returns the node associated with an id, or nullptr if no node
            // with that id (the node with that id was removed and the id has
            // not yet been re-used). *this owns the returned instance.
            // REQUIRES: 0 <= id < get_max_node_id().
            std::shared_ptr<GNode> find_node_id(size_t id) const { return m_nodes[id]; }
            // Adds an edge that connects the xth output of `source` to the yth input of
            // `dest` and returns it. Does not update dest's NodeDef.
            const std::shared_ptr<Edge>
                add_edge(std::shared_ptr<GNode> source, int x, std::shared_ptr<GNode> dest, int y);

            // Removes edge from the graph. Does not update the destination node's
            // NodeDef.
            // REQUIRES: The edge must exist.
            void remove_edge(const std::shared_ptr<Edge> edge);

            // Updates the input to a node.  The existing edge to `dst` is removed and an
            // edge from `new_src` to `dst` is created. The NodeDef associated with `dst`
            // is also updated.
            //Status UpdateEdge(std::shared_ptr<Node> new_src, int new_src_index, std::shared_ptr<Node> dst, int dst_index);

            // The number of live edges in the graph.
            //
            // Because edges can be removed from the graph, get_edge_size() is often
            // smaller than get_max_edge_id(). If one needs to create an array of
            // edges indexed by edge ids, get_max_edge_id() should be used as the
            // array's size.
            size_t get_edge_size() const { return m_edge_size; }
            // Returns one more than the maximum id assigned to any edge.
            size_t get_max_edge_id() const { return m_edges.size(); }
            // Returns the Edge associated with an id, or nullptr if no edge
            // with that id (the node with that id was removed and the id has
            // not yet been re-used). *this owns the returned instance.
            // REQUIRES: 0 <= id < get_max_node_id().
            const std::shared_ptr<Edge> find_edge_id(size_t id) const { return m_edges[id]; }
            // TODO: to be discussed
            std::vector<std::shared_ptr<GNode>> get_output_nodes() const { return m_output_nodes; }
            void set_output_nodes(const std::vector<std::shared_ptr<GNode>> nodes)
            {
                m_output_nodes = nodes;
            }
            void set_default_output_nodes();

        private:
            // Map from node ids to allocated nodes.  nodes_[id] may be nullptr if
            // the node with that id was removed from the graph.
            std::vector<std::shared_ptr<GNode>> m_nodes;

            // Number of nodes alive.
            size_t m_node_size = 0;

            // Map from edge ids to allocated edges.  m_edges[id] may be nullptr if
            // the edge with that id was removed from the graph.
            std::vector<std::shared_ptr<Edge>> m_edges;

            // The number of entries in m_edges that are not nullptr.
            size_t m_edge_size = 0;

            // Allocated but free nodes and edges.
            std::vector<std::shared_ptr<GNode>> m_free_nodes;
            std::vector<std::shared_ptr<Edge>> m_free_edges;

            // TODO: Output nodes of this graph
            std::vector<std::shared_ptr<GNode>> m_output_nodes;

            // For generating unique names.
            int name_counter_ = 0;
        };

        inline bool Edge::is_control_edge() const
        {
            // Note that if either src_output_ or dst_input_ is kControlSlot,
            // so is the other one (add_edge checks this).
            return m_src_output == Graph::kControlSlot;
        }
    }
}