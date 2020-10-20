// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "reduce_fusion_pass.hpp"
#include <queue>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

DEFINE_bool(freduce_fusion, false, "Enable reduce-range based kernel fusion.");
DEFINE_int32(freduce_range, 512, "Reduce range.");

REGISTER_OP(Matched_Pattern)
    // .attr<nnfusion::op::OpConfig::any>("out_shape")
    .infershape([](std::shared_ptr<GNode> gnode) -> void {
        // auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        // Shape out_shape = op->localOpConfig.getRoot()["out_shape"];
        // gnode->set_output_type_and_shape(0, element::f32, out_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Fused>(curr->get_op_ptr());
        return _op->get_fused_ir2() + _op->get_plan_rule();
    });
namespace
{
    struct FuseGroup;  
    struct TaggedNode
    {
        TaggedNode()
            : node(nullptr)
            , elem_group(nullptr)
            , ready_inputs(0)
            , visited(false)
        {
        }

        std::shared_ptr<GNode> node;

        std::shared_ptr<FuseGroup> elem_group;
        size_t ready_inputs;
        bool visited;
    };

    struct FuseGroup
    {
        FuseGroup() {}
        std::vector<size_t> nodes;
    };
}

class ReduceFusionOptimizer
{
public:
    ReduceFusionOptimizer(std::shared_ptr<Graph> g)
        : m_graph(g)
    {
        m_nodes.resize(m_graph->get_max_node_id());
        ELEM_GROUP_NODEID = m_nodes.size();
    };

    bool Optimize()
    {
        std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> fuse_groups =
            ExtractFusionGroups();
        Substitution(fuse_groups);
        return true;
    }

private:
    size_t ELEM_GROUP_NODEID;
    size_t group_size(std::shared_ptr<FuseGroup> group)
    {
        if (!group)
            return 0;
        size_t num_nodes = 0;
        for (auto id : group->nodes)
        {
            NNFUSION_CHECK(id < m_nodes.size());
            auto tn = m_nodes[id];
            if (id >= ELEM_GROUP_NODEID && tn->elem_group)
            {
                num_nodes += tn->elem_group->nodes.size();
            }
            else
            {
                num_nodes++;
            }
        }
        return num_nodes;
    }

    float compute_shared_memory(std::shared_ptr<nnfusion::op::Op> op)
    {
        if (op->get_shared_memory().empty())
            return -1.0f;

        float shared_memory = 1;
        for (float i : op->get_shared_memory())
            shared_memory *= i;

        return shared_memory;
    }

    std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> ExtractFusionGroups()
    {
        std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> groups =
            std::make_shared<std::vector<std::shared_ptr<FuseGroup>>>();
        std::queue<size_t> ready;
        std::deque<size_t> elem_ready;
        enum WorkState
        {
            PROCESS_READY_NODE,
            PROCESS_FUSIABLE_NODE,
            PROCESS_ELEM_NODE,
            WORK_DONE
        };

        WorkState state = PROCESS_READY_NODE;
        std::shared_ptr<FuseGroup> cur_group = nullptr;
        std::shared_ptr<FuseGroup> cur_elemgroup = nullptr;
        for (auto node : m_graph->get_ordered_ops())
        {
            size_t id = node->get_id();
            m_nodes[id] = std::make_shared<TaggedNode>();
            m_nodes[id]->node = node;
            if (!(m_nodes[id]->visited) &&
                (m_nodes[id]->ready_inputs == node->get_in_edges().size()))
            {
                ready.push(id);
            }
        }
        while (state != WORK_DONE)
        {
            size_t node_id = 0;
            std::shared_ptr<TaggedNode> tn = nullptr;
            switch (state)
            {
            // Process the nodes in ready queue
            case PROCESS_READY_NODE:
            {
                if (ready.empty())
                {
                    state = (elem_ready.empty()) ? WORK_DONE : PROCESS_ELEM_NODE;
                    break;
                }

                node_id = ready.front();
                ready.pop();
                tn = m_nodes[node_id];
                break;
            }
            // Process the nodes in elem_ready queue
            case PROCESS_ELEM_NODE:
            {
                auto AppendElementGroup = [&]() {
                    if (cur_elemgroup && cur_elemgroup->nodes.size() > 0)
                    {
                        // append cur_elemgroup to cur_group
                        if (!cur_group)
                        {
                            cur_group = std::make_shared<FuseGroup>();
                        }
                        auto new_tn = std::make_shared<TaggedNode>();
                        new_tn->elem_group = cur_elemgroup;
                        int new_id = m_nodes.size();
                        m_nodes.push_back(new_tn);
                        cur_group->nodes.push_back(new_id);
                        cur_elemgroup = nullptr;
                    }
                };
                if (elem_ready.empty())
                {
                    AppendElementGroup();
                    // Close the cur_group
                    if (cur_group && cur_group->nodes.size() > 0)
                    {
                        if (group_size(cur_group) > 1)
                        {
                            groups->push_back(cur_group);
                        }
                        cur_group = nullptr;
                    }
                    state = PROCESS_READY_NODE;
                    break;
                }
                node_id = elem_ready.front();
                elem_ready.pop_front();
                tn = m_nodes[node_id];
                if (cur_elemgroup)
                {
                    AppendElementGroup();
                }
                if (!cur_elemgroup)
                {
                    cur_elemgroup = std::make_shared<FuseGroup>();
                }
                cur_elemgroup->nodes.push_back(node_id);
                break;
            }

            // Do nothing
            case WORK_DONE: { break;
            }
            } // switch
            if (tn)
            {
                tn->visited = true;
                for (auto edge : tn->node->get_out_edges())
                {
                    auto dst = m_nodes[edge->get_dst()->get_id()];
                    if (!dst)
                        continue;
                    dst->ready_inputs++;
                    if (!(dst->visited) && (dst->ready_inputs >= dst->node->get_in_edges().size()))
                    {
                        auto op = dst->node->get_op_ptr();
                        auto sm = compute_shared_memory(op);
                        if (sm > 0 && sm < FLAGS_freduce_range)
                        {
                            elem_ready.push_front(dst->node->get_id());
                        }
                        else
                        {
                            ready.push(dst->node->get_id());
                        }
                    }
                }
            }
        } // while
        return groups;
    }

    void Substitution(std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> matched)
    {
        for (auto group : *matched)
        {
            for (auto id : group->nodes)
            {
                NNFUSION_CHECK(id < m_nodes.size());
                auto tn = m_nodes[id];
                if (id >= ELEM_GROUP_NODEID && tn->elem_group && tn->elem_group->nodes.size() > 1)
                {
                    // nnfusion::op::OpConfig::any op_config;
                    // op_config["out_shape"] = Shape(matched.back()->node->get_output_shape(0));
                    auto subs_op =
                        std::make_shared<nnfusion::op::Fused>("Matched_Pattern", "Matched_Pattern");
                    GNodeVector empty_inputs;
                    auto subs_node = std::make_shared<GNode>(subs_op, empty_inputs);

                    m_graph->add_node(subs_node);

                    std::unordered_set<std::shared_ptr<GNode>> internal_nodes;
                    for (auto elem_id : tn->elem_group->nodes)
                    {
                        auto node = m_nodes[elem_id]->node;
                        internal_nodes.insert(node);
                    }

                    std::unordered_set<std::shared_ptr<GNode>> matched_set;
                    for (auto elem_id : tn->elem_group->nodes)
                    {
                        auto node = m_nodes[elem_id]->node;
                        // Add non-control-edges as inputs of fused node
                        auto next_input_id_base = subs_node->get_input_size();
                        for (auto in_id = 0; in_id < node->get_input_size(); ++in_id)
                        {
                            auto& in_edge = node->get_in_edge(in_id);
                            if (internal_nodes.find(in_edge->get_src()) == internal_nodes.end())
                            {
                                auto input_id = next_input_id_base++;
                                subs_node->set_input(
                                    input_id, node->get_inputs().at(in_edge->get_dst_input()));
                                m_graph->add_edge(in_edge->get_src(),
                                                  in_edge->get_src_output(),
                                                  subs_node,
                                                  input_id);
                            }
                        }
                        // Add control-edges as inputs of fused node
                        for (const auto& in_edge : node->get_in_edges())
                        {
                            if (in_edge->is_control_edge())
                            {
                                m_graph->add_edge(in_edge->get_src(),
                                                  in_edge->get_src_output(),
                                                  subs_node,
                                                  Graph::kControlSlot);
                            }
                        }
                    }

                    auto next_output_id = 0;
                    for (auto elem_id : tn->elem_group->nodes)
                    {
                        auto node = m_nodes[elem_id]->node;
                        for (int out_id = 0; out_id < node->get_output_size(); ++out_id)
                        {
                            bool has_output = false;
                            auto out_edges = node->get_output_users(out_id);
                            for (auto out_edge : out_edges)
                            {
                                auto out_node = out_edge->get_dst();
                                if (internal_nodes.find(out_node) != internal_nodes.end())
                                    continue;
                                if (!has_output)
                                {
                                    has_output = true;
                                    subs_node->set_output(
                                        next_output_id++,
                                        node->get_outputs().at(out_edge->get_src_output()));
                                }
                                m_graph->add_edge(subs_node,
                                                  next_output_id - 1,
                                                  out_edge->get_dst(),
                                                  out_edge->get_dst_input());
                            }
                        }
                    }

                    std::vector<std::shared_ptr<graph::GNode>> matched_nodes;
                    for (auto elem_id : tn->elem_group->nodes)
                    {
                        auto node = m_nodes[elem_id]->node;
                        matched_nodes.push_back(node);
                    }
                    subs_op->register_ir2(matched_nodes);

                    for (auto node : internal_nodes)
                    {
                        m_graph->remove_node(node);
                    }
                }
            }
        }
    }

    std::shared_ptr<Graph> m_graph;
    std::vector<std::shared_ptr<TaggedNode>> m_nodes;
};

bool ReduceFusionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    if (FLAGS_freduce_fusion)
    {
        ReduceFusionOptimizer optimizer(graph);
        return optimizer.Optimize();
    }
    return true;
}