// Microsoft (c) 2019, NNFusion Team

#pragma once

#include <atomic>
#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/node.hpp"

namespace nnfusion
{
    namespace graph
    {
        class Edge;

        /// Nodes are the backbone of the graph of Value dataflow. Every node has
        /// zero or more nodes as arguments and one value, which is either a tensor
        /// view or a (possibly empty) tuple of values.
        class GNode : public std::enable_shared_from_this<GNode>
        {
        protected:
            GNode(const std::shared_ptr<ngraph::Node> op_ptr);

        public:
            GNode();
            ~GNode();
            void initialize(const std::shared_ptr<ngraph::Node> op_ptr);
            size_t get_instance_id() const { return m_instance_id; }
            size_t get_id() const { return m_id; }
            size_t set_id(size_t id) { m_id = id; }
            /// The class name, must not contain spaces
            std::string get_op_type() const { return m_op_type; }
            std::shared_ptr<ngraph::Node> get_op_ptr() const { return m_op_ptr; }
            const std::string& get_unique_name() const;
            const std::string& get_name() const;
            void set_name(const std::string& name);

            /// Get in edges
            const std::set<std::shared_ptr<Edge>>& get_in_edges() const;
            void add_in_edge(std::shared_ptr<Edge> edge);
            void remove_in_edge(std::shared_ptr<Edge> edge) { m_in_edges.erase(edge); }
            /// Get out edges
            const std::set<std::shared_ptr<Edge>>& get_out_edges() const;
            void add_out_edge(std::shared_ptr<Edge> edge);
            void remove_out_edge(std::shared_ptr<Edge> edge) { m_out_edges.erase(edge); }
            size_t get_output_size() const { return m_out_edges.size(); }
            void Clear();

            bool is_constant() const { return m_op_ptr->is_constant(); }
            /// Use instance ids for comparison instead of memory addresses to improve determinism
            bool operator<(const GNode& other) const { return m_instance_id < other.m_instance_id; }
        protected:
            size_t m_id; // m_id is for graph, the index in graph m_nodes
            size_t m_instance_id;
            static std::atomic<size_t> m_next_instance_id;
            std::string m_name;
            const std::string m_unique_name;

            std::string m_op_type;
            std::shared_ptr<ngraph::Node> m_op_ptr;

            std::set<std::shared_ptr<Edge>> m_in_edges;
            std::set<std::shared_ptr<Edge>> m_out_edges;
        };
    }
}
