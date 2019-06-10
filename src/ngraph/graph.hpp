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

#pragma once

#include <memory>
#include <vector>

namespace ngraph
{
    class Node;

    class Edge
    {
    public:
        std::shared_ptr<Node> get_src() const { return m_src; }
        std::shared_ptr<Node> get_dst() const { return m_dst; }
        size_t get_id() const { return m_id; }
        // Return the index of the source output that produces the data
        // carried by this edge.  The special value kControlSlot is used
        // for control dependencies.
        int get_src_output() const { return m_src_output; }
        // Return the index of the destination input that consumes the data
        // carried by this edge.  The special value kControlSlot is used
        // for control dependencies.
        int get_dst_input() const { return m_dst_input; }
        // Return true iff this is an edge that indicates a control-flow
        // (as opposed to a data-flow) dependency.
        bool IsControlEdge() const;

        std::string DebugString() const;

    private:
        friend class Graph;
        std::shared_ptr<Node> m_src;
        std::shared_ptr<Node> m_dst;
        size_t m_id;
        int m_src_output;
        int m_dst_input;
    };

    // Thread compatible but not thread safe.
    class Graph
    {
    public:
        // Constructs a graph with a single SOURCE (always id kSourceId) and a
        // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.
        explicit Graph();

        ~Graph();

        static const int kControlSlot;

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