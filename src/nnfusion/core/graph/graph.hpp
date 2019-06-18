// Microsoft (c) 2019, NNFusion Team

#pragma once

#include <memory>
#include <vector>
#include "node.hpp"
#include "edge.hpp"

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
            void AddNode(std::shared_ptr<Node> node);

            // Removes a node from this graph, including all edges from or to it.
            // *node should not be accessed after calling this function.
            // REQUIRES: node->IsOp()
            void RemoveNode(std::shared_ptr<Node> node);

            // The number of live nodes in the graph.
            //
            // Because nodes can be removed from the graph, num_nodes() is often
            // smaller than num_node_ids(). If one needs to create an array of
            // nodes indexed by node ids, num_node_ids() should be used as the
            // array's size.
            size_t get_num_nodes() const { return m_num_nodes; }
            // Returns one more than the maximum id assigned to any node.
            size_t get_num_node_ids() const { return m_nodes.size(); }
            // Returns the node associated with an id, or nullptr if no node
            // with that id (the node with that id was removed and the id has
            // not yet been re-used). *this owns the returned instance.
            // REQUIRES: 0 <= id < num_node_ids().
            std::shared_ptr<Node> FindNodeId(size_t id) const { return m_nodes[id]; }
            // Adds an edge that connects the xth output of `source` to the yth input of
            // `dest` and returns it. Does not update dest's NodeDef.
            const std::shared_ptr<Edge>
                AddEdge(std::shared_ptr<Node> source, int x, std::shared_ptr<Node> dest, int y);

            // Removes edge from the graph. Does not update the destination node's
            // NodeDef.
            // REQUIRES: The edge must exist.
            void RemoveEdge(const std::shared_ptr<Edge> edge);

            // Updates the input to a node.  The existing edge to `dst` is removed and an
            // edge from `new_src` to `dst` is created. The NodeDef associated with `dst`
            // is also updated.
            //Status UpdateEdge(std::shared_ptr<Node> new_src, int new_src_index, std::shared_ptr<Node> dst, int dst_index);

            // The number of live edges in the graph.
            //
            // Because edges can be removed from the graph, num_edges() is often
            // smaller than num_edge_ids(). If one needs to create an array of
            // edges indexed by edge ids, num_edge_ids() should be used as the
            // array's size.
            size_t get_num_edges() const { return m_num_edges; }
            // Returns one more than the maximum id assigned to any edge.
            size_t get_num_edge_ids() const { return m_edges.size(); }
            // Returns the Edge associated with an id, or nullptr if no edge
            // with that id (the node with that id was removed and the id has
            // not yet been re-used). *this owns the returned instance.
            // REQUIRES: 0 <= id < num_node_ids().
            const std::shared_ptr<Edge> FindEdgeId(size_t id) const { return m_edges[id]; }
        private:
            // Map from node ids to allocated nodes.  nodes_[id] may be nullptr if
            // the node with that id was removed from the graph.
            std::vector<std::shared_ptr<Node>> m_nodes;

            // Number of nodes alive.
            size_t m_num_nodes = 0;

            // Map from edge ids to allocated edges.  m_edges[id] may be nullptr if
            // the edge with that id was removed from the graph.
            std::vector<std::shared_ptr<Edge>> m_edges;

            // The number of entries in m_edges that are not nullptr.
            size_t m_num_edges = 0;

            // Allocated but free nodes and edges.
            std::vector<std::shared_ptr<Node>> m_free_nodes;
            std::vector<std::shared_ptr<Edge>> m_free_edges;
        };

        inline bool Edge::IsControlEdge() const
        {
            // Note that if either src_output_ or dst_input_ is kControlSlot,
            // so is the other one (AddEdge checks this).
            return m_src_output == Graph::kControlSlot;
        }
    }
}