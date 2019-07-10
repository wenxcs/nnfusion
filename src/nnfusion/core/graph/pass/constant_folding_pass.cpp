// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "constant_folding_pass.hpp"
#include "../graph.hpp"
#include "../node.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::graph::pass;

// This generator type is used to generate a name for the newly folded node
// based on the node's old name.
using ConstantFoldNameGenerator =
    std::function<std::string(std::shared_ptr<Graph>, std::string old_name)>;

// Comparator for two nodes. This is used in order to get a stable ording.
using NodeComparator =
    std::function<bool(const std::shared_ptr<Node>, const std::shared_ptr<Node>)>;

// Compares two node based on their ids.
struct NodeComparatorID
{
    bool operator()(const std::shared_ptr<Node> n1, const std::shared_ptr<Node> n2) const
    {
        return n1->get_id() < n2->get_id();
    }
};

// Compare two nodes based on their names.
struct NodeComparatorName
{
    bool operator()(const std::shared_ptr<Node> n1, const std::shared_ptr<Node> n2) const
    {
        return n1->get_name() < n2->get_name();
    }
};

long long UniqueConstantId()
{
    static std::atomic_int_fast64_t unique_constant_id;
    return unique_constant_id.fetch_add(1);
}

typedef std::pair<std::shared_ptr<Node>, int> NodeAndOutput;

void ReverseDFS(const std::shared_ptr<Graph>& graph,
                const std::vector<std::shared_ptr<Node>>& start,
                const std::function<void(std::shared_ptr<Node>)>& enter,
                const std::function<void(std::shared_ptr<Node>)>& leave,
                const NodeComparator& stable_comparator)
{
    // Stack of work to do.
    struct Work
    {
        std::shared_ptr<Node> node;
        bool leave; // Are we entering or leaving n?
    };
    std::vector<Work> stack(start.size());
    for (int i = 0; i < start.size(); ++i)
    {
        stack[i] = Work{start[i], false};
    }

    std::vector<bool> visited(graph->num_node_ids(), false);
    while (!stack.empty())
    {
        Work w = stack.back();
        stack.pop_back();

        std::shared_ptr<Node> n = w.node;
        if (w.leave)
        {
            leave(n);
            continue;
        }

        if (visited[n->get_id()])
        {
            continue;
        }
        visited[n->get_id()] = true;
        if (enter)
        {
            enter(n);
        }

        // Arrange to call leave(n) when all done with descendants.
        if (leave)
        {
            stack.push_back(Work{n, true});
        }

        auto add_work = [&visited, &stack](std::shared_ptr<Node> out) {
            if (!visited[out->get_id()])
            {
                // Note; we must not mark as visited until we actually process it.
                stack.push_back(Work{out, false});
            }
        };

        if (stable_comparator)
        {
            std::vector<std::shared_ptr<Node>> nodes_sorted;
            for (auto in_edge : n->get_in_edges())
            {
                nodes_sorted.emplace_back(in_edge->get_src());
            }
            std::sort(nodes_sorted.begin(), nodes_sorted.end(), stable_comparator);
            for (auto in : nodes_sorted)
            {
                add_work(in);
            }
        }
        else
        {
            for (auto in_edge : n->get_in_edges())
            {
                add_work(in_edge->get_src());
            }
        }
    }
}

// Returns true if n can be evaluated as constant.
bool IsConstantFoldable(const std::shared_ptr<Node> node)
{
    if (node->is_constant())
    {
        return true;
    }
    // TODO check if node in "CPU" is supported
    /*
    if (n->op_def().is_stateful()) {
    return false;
    }
    if (consider && !consider(n)) {
        return false;
    }
    if (n->IsControlFlow() || n->IsSend() || n->IsRecv()) {
        return false;
    }
    // TODO(yuanbyu): For now disable these session handle operations.
    if (n->IsGetSessionHandle() || n->IsGetSessionTensor() ||
        n->IsDeleteSessionTensor()) {
        return false;
    }
    if (n->IsSource()) {
        return false;
    }
    if (n->IsSink()) {
        return false;
    }
    if (n->IsFakeParam()) {
        return false;
    }
   */

    // Since constant-folding runs on the CPU, do not attempt to constant-fold
    // operators that have no CPU kernel. Also implies that we will not
    // constant-fold functions.
    // TODO(phawkins): allow constant-folding for functions; functions may
    // be arbitrarily expensive to execute.
    //if (!KernelDefAvailable(DeviceType(DEVICE_CPU), n->def())) {
    //  return false;
    //}
    return true;
}

// If n is eligible for constant-folding, adds it to nodes, and places its
// control dependencies and those transitively of its constant-foldable inputs
// into constant_control_deps. If n is a constant-foldable shape node (Shape,
// ShapeN, Rank, or Size), also puts its outputs into shape_replacement_map.
void ConsiderConstantFoldableNode(
    std::shared_ptr<Node> node,
    std::vector<std::shared_ptr<Node>>* constant_foldable_nodes,
    std::unordered_map<std::shared_ptr<Node>, std::set<std::shared_ptr<Node>>>*
        constant_control_deps,
    bool* internal_node_inserted)
{
    if (IsConstantFoldable(node))
    {
        // A node is constant provided all of its non-control incoming Tensors come
        // from constant nodes, or it's a shape Op with statically known inputs in
        // which case it is placed in shape_replacement_map.
        //
        // We allow control dependencies from non-constant nodes to constant nodes,
        // but to preserve the graph structure we must transfer the control
        // dependency onto any constant replacement.
        bool all_parents_constant = true;
        for (auto in_edge : node->get_in_edges())
        {
            // Allows non-constant -> constant control edges.
            //TODO:??????
            if (!in_edge->is_control_edge() &&
                constant_control_deps->count(in_edge->get_src()) == 0)
            {
                all_parents_constant = false;
                break;
            }
        }
        if (all_parents_constant)
        {
            std::set<std::shared_ptr<Node>>& control_deps = (*constant_control_deps)[node];
            for (auto in_edge : node->get_in_edges())
            {
                if (constant_control_deps->count(in_edge->get_src()) == 0)
                {
                    // This branch is taken if the incoming edge is a control dependency,
                    // in which case we want to add it to the dependencies being
                    // accumulated for this node, or the incoming edge is not
                    // constant. The latter may happen when n is a shape node and the
                    // source has known shape. In that case add a control dependency from
                    // the source node, since there was previously a data dependency and
                    // we want to preserve sequencing constraints.

                    //TDOO
                    //if (!in_edge->get_src()->IsSource())
                    {
                        control_deps.insert(in_edge->get_src());
                    }
                }
                else
                {
                    // If the parent has been accumulating control dependencies, add all
                    // of its transitive control deps.
                    std::set<std::shared_ptr<Node>>& parent_deps =
                        (*constant_control_deps)[in_edge->get_src()];
                    control_deps.insert(parent_deps.begin(), parent_deps.end());
                }
            }
            constant_foldable_nodes->push_back(node);
            if (!node->is_constant())
            {
                *internal_node_inserted = true;
            }
        }
    }
}

// Adds node to constant_graph which is being built up for subsequent evaluation of
// constant propagation. node_map is the mapping of nodes in the original graph
// to nodes in the constant graph.
void AddNodeToConstantGraph(
    std::shared_ptr<Node> node,
    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>>* node_map,
    std::shared_ptr<Graph> constant_graph)
{
    std::shared_ptr<Node>& new_node = (*node_map)[node];
    // todo: copy node is not implemented yet
    new_node = constant_graph->copy_node(node);
    for (const auto in_edge : node->get_in_edges())
    {
        // Don't copy control edges to the constant graph.
        if (!in_edge->is_control_edge())
        {
            auto src_node = in_edge->get_src();
            auto it = node_map->find(src_node);
            if (it == node_map->end())
            {
                std::cerr << node->get_friendly_name() << " <-" << src_node->get_friendly_name();
                return;
            }

            constant_graph->add_edge(
                it->second, in_edge->get_src_output(), new_node, in_edge->get_dst_input());
        }
    }
}

// Given the constant foldable nodes in 'nodes', returns a new graph 'g'. 'g'
// will contain copies of the nodes in 'nodes'. In addition, if there is an edge
// going from a node 'n' in 'nodes' to another node in 'orig_graph' but not in
// 'nodes', then 'tensors_to_fetch' will contain the mapping from the
// corresponding copy of 'n' and the edge number in 'g' to 'n'.
std::shared_ptr<Graph>
    GetConstantGraph(const std::shared_ptr<Graph>& orig_graph,
                     const std::vector<std::shared_ptr<Node>>& nodes,
                     std::map<NodeAndOutput, std::shared_ptr<Node>>* tensors_to_fetch,
                     const ConstantFoldNameGenerator& generate_new_name)
{
    std::shared_ptr<Graph> constant_graph = std::make_shared<Graph>();
    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>> node_map;

    for (auto n : nodes)
    {
        AddNodeToConstantGraph(n, &node_map, constant_graph);
    }

    for (auto const& added_nodes : node_map)
    {
        for (const auto out_edge : added_nodes.first->get_out_edges())
        {
            if (node_map.count(out_edge->get_dst()) == 0)
            {
                if (out_edge->is_control_edge())
                {
                    continue;
                }

                tensors_to_fetch->insert(
                    {{added_nodes.second, out_edge->get_src_output()}, added_nodes.first});
            }
        }
    }

    return constant_graph;
}

bool ConstantFoldingPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    // DFS
    // TODO: to be removed if output_nodes are specifed in graph
    graph->set_default_output_nodes();

    auto outputs = graph->get_output_nodes();

    std::vector<std::shared_ptr<Node>> constant_foldable_nodes;
    std::unordered_map<std::shared_ptr<Node>, std::set<std::shared_ptr<Node>>>
        constant_control_deps;
    bool internal_node_inserted = false;

    ReverseDFS(graph,
               outputs,
               nullptr,
               [&constant_foldable_nodes, &constant_control_deps, &internal_node_inserted](
                   std::shared_ptr<Node> node) {
                   ConsiderConstantFoldableNode(node,
                                                &constant_foldable_nodes,
                                                &constant_control_deps,
                                                &internal_node_inserted);
               },
               NodeComparatorName());

    // If we have inserted just leaf level nodes, then there is nothing to fold.
    if (!internal_node_inserted)
    {
        constant_foldable_nodes.clear();
        constant_control_deps.clear();
    }

    if (constant_foldable_nodes.empty())
    {
        //VLOG(1) << "No constant foldable nodes found";
        //*was_mutated = false;
        // This is not an error, so return the status as OK.
        return true;
    }

    // generate constant graph
    ConstantFoldNameGenerator generate_new_name = [](std::shared_ptr<Graph> graph,
                                                     std::string old_name) {
        return graph->new_name(old_name).append("__cf__").append(
            std::to_string(UniqueConstantId()));
    };

    std::map<NodeAndOutput, std::shared_ptr<Node>> tensors_to_fetch;
    // todo: unique_ptr<Graph> ??
    auto constant_graph =
        GetConstantGraph(graph, constant_foldable_nodes, &tensors_to_fetch, generate_new_name);

    if (tensors_to_fetch.empty())
    {
        //VLOG(1) << "No constant nodes found that feed into the original graph.";
        //*was_mutated = false;
        // This is not an error, so return the status as OK.
        return true;
    }

    std::vector<std::string> tensors_to_fetch_names;
    //old node : output src
    std::vector<NodeAndOutput> tensors_to_replace;
    // Sorting the nodes based on the name gives us a stable ordering between runs
    // for the same graph.
    std::vector<std::pair<NodeAndOutput, std::shared_ptr<Node>>> tensors_to_fetch_sorted(
        tensors_to_fetch.begin(), tensors_to_fetch.end());
    std::sort(
        tensors_to_fetch_sorted.begin(),
        tensors_to_fetch_sorted.end(),
        [](const std::pair<NodeAndOutput, std::shared_ptr<Node>>& n1, const std::pair<NodeAndOutput, std::shared_ptr<Node>>& n2) {
            return n1.first.first->get_name() < n2.first.first->get_name();
        });
    for (auto n : tensors_to_fetch_sorted)
    {
        tensors_to_fetch_names.push_back(n.first.first->get_name() + ":" +
                                         std::to_string(n.first.second));
        tensors_to_replace.push_back({n.second, n.first.second});
    }

    // execute
    /*
        Shape shape{4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("INTERPRETER");

    shared_ptr<runtime::interpreter::INTBackend> ibackend =
        static_pointer_cast<runtime::interpreter::INTBackend>(backend);

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, NAN, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 1, 8});
    auto result = backend->create_tensor(element::f32, shape);

    ibackend->set_nan_check(f, true);
    EXPECT_ANY_THROW(ibackend->call_with_validate(f, {result}, {a, b}));
     */

    // new graph
    return true;
}