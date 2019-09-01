// Microsoft (c) 2019, NNFusion Team

#pragma once

#include <memory>
#include <vector>
#include "gedge.hpp"
#include "gnode.hpp"
#include "ngraph/function.hpp"
#include "ngraph/op/parameter_vector.hpp"

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

            Graph(const std::string& name = "");

            Graph(const std::shared_ptr<ngraph::Function>& func, const std::string& name = "");

            ~Graph();

            static const int kControlSlot = -1;

            const std::string& get_friendly_name() const;
            const std::string& get_name() const;
            void set_name(const std::string& name);

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

            std::vector<std::shared_ptr<GNode>> get_nodes();
            std::vector<std::shared_ptr<GNode>> get_ordered_ops(bool include_control_deps = true);
            std::shared_ptr<GNode> find_node_id(size_t id) const { return m_nodes[id]; }
            // Adds an edge that connects the xth output of `source` to the yth input of
            // `dest` and returns it. Does not update dest's NodeDef.
            const std::shared_ptr<Edge>
                add_edge(std::shared_ptr<GNode> source, int x, std::shared_ptr<GNode> dest, int y);

            bool
                find_edge(std::shared_ptr<GNode> source, int x, std::shared_ptr<GNode> dest, int y);
            const std::shared_ptr<Edge> add_control_edge(std::shared_ptr<GNode> source,
                                                         std::shared_ptr<GNode> dest,
                                                         bool allow_duplicates = false);
            // Removes edge from the graph. Does not update the destination node's
            // NodeDef.
            // REQUIRES: The edge must exist.
            void remove_edge(const std::shared_ptr<Edge> edge);
            void remove_control_edge(const std::shared_ptr<Edge> edge);
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
            std::vector<std::shared_ptr<GNode>> get_outputs();

            void set_outputs(std::vector<std::shared_ptr<GNode>> outputs);
            void set_default_outputs();
            const size_t get_output_size();
            /// Return the op that generates output i
            const std::shared_ptr<GNode> get_output_op(size_t i);

            std::vector<std::shared_ptr<GNode>> get_parameters();
            void set_default_parameters();

            size_t get_temporary_pool_size();
            void set_temporary_pool_size(size_t);

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
            std::vector<std::shared_ptr<GNode>> m_parameters;
            // For generating unique names.
            int name_counter_ = 0;

            static std::atomic<size_t> m_next_instance_id;
            size_t m_instance_id;
            std::string m_name;
            const std::string m_unique_name;

            size_t m_temporary_pool_size;
        };

        inline bool Edge::is_control_edge() const
        {
            // Note that if either src_output_ or dst_input_ is kControlSlot,
            // so is the other one (add_edge checks this).
            return m_src_output == Graph::kControlSlot;
        }
    }
}
