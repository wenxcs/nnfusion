// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "kernel_fusion_pass.hpp"
#include <queue>
#include "../gnode.hpp"
#include "../graph.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::graph::pass;
using namespace ngraph::op::util;

const static int DEFAULT_GROUP_ID = -1;

struct FuseGroup;
struct TaggedNode
{
    TaggedNode()
        : node(nullptr)
        , group_id(DEFAULT_GROUP_ID)
        , elem_group(nullptr)
        , ready_inputs(0)
        , visited(false)
    {
    }

    std::shared_ptr<GNode> node;
    int group_id;
    std::shared_ptr<FuseGroup> elem_group;
    size_t ready_inputs;
    bool visited;
};

struct FuseGroup
{
    FuseGroup(int g_id = DEFAULT_GROUP_ID)
        : id(g_id)
    {
    }
    int id;

    // <nodeid, <src_output_idx, in_slot_id>>
    std::unordered_map<std::shared_ptr<GNode>, std::unordered_map<int, int>> inputs;
    std::unordered_map<std::shared_ptr<GNode>, std::unordered_map<int, int>> consts;
    // <nodeid, <dst_input_idx, out_slot_id>>
    std::unordered_map<std::shared_ptr<GNode>, std::unordered_map<int, int>> outputs;

    size_t num_inputs;
    size_t num_consts;
    size_t num_outputs;

    std::vector<size_t> nodes;
};

class KernelFuseOptimizer
{
public:
    KernelFuseOptimizer(std::shared_ptr<Graph> g)
        : m_graph(g)
    {
        m_nodes.resize(m_graph->get_max_node_id(), TaggedNode());
        ELEM_GROUP_NODEID = m_nodes.size();
        RegisterFusionOps();
    }

    bool Optimize()
    {
        std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> fuse_groups =
            ExtractFusionGroups();
        if (fuse_groups != nullptr && fuse_groups->size() > 0)
        {
            TagFusionGroupsOnGraph(fuse_groups);
            return true;
        }
        return true;
    }

private:
    void RegisterFusionOps()
    {
        static const std::vector<std::string> fuseop_list = {};
        //{"MatMul", "Split", "Concat", "ConcatV2", "Reshape"};

        static const std::vector<std::string> blacklist = {"Softmax"};

        fusion_ops.insert(fuseop_list.begin(), fuseop_list.end());
        op_blacklist.insert(blacklist.begin(), blacklist.end());

        // RegisterFusionOpFilters();
        host_inputs = {
            {"Split", 0}, {"Concat", 0}, {"ConcatV2", -1}, {"Reshape", 1}, {"ExpandDims", 1}};
    }

    void AddNodeToReadyQueues(std::shared_ptr<GNode> node,
                              std::queue<size_t>& ready,
                              std::deque<size_t>& fuse_ready,
                              std::deque<size_t>& elem_ready)
    {
        std::shared_ptr<ngraph::Node> op = node->get_op_ptr();

        if (op_blacklist.count(node->get_op_type()) > 0)
        {
            ready.push(node->get_id());
            return;
        }

        if (std::dynamic_pointer_cast<BinaryElementwiseArithmetic>(op) ||
            std::dynamic_pointer_cast<UnaryElementwiseArithmetic>(op))
        {
            elem_ready.push_front(node->get_id());
        }
        else if (fusion_ops.find(node->get_op_type()) != fusion_ops.end())
        {
            fuse_ready.push_front(node->get_id());
        }
        else
        {
            ready.push(node->get_id());
        }
    }

    size_t group_size(std::shared_ptr<FuseGroup> group)
    {
        if (!group)
            return 0;
        size_t num_nodes = 0;
        for (auto id : group->nodes)
        {
            enforce(id < m_nodes.size());
            TaggedNode& tn = m_nodes[id];
            if (id >= ELEM_GROUP_NODEID && tn.elem_group)
            {
                num_nodes += tn.elem_group->nodes.size();
            }
            else
            {
                num_nodes++;
            }
        }
        return num_nodes;
    }

    std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> ExtractFusionGroups()
    {
        std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> groups =
            std::make_shared<std::vector<std::shared_ptr<FuseGroup>>>();
        std::queue<size_t> ready;
        std::deque<size_t> fuse_ready;
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

        for (auto node : m_graph->get_nodes())
        {
            size_t id = node->get_id();
            m_nodes[node->get_id()].node = node;
            if (!m_nodes[id].visited && m_nodes[id].ready_inputs == node->get_in_edges().size())
            {
                ready.push(id);
            }
        }

        while (state != WORK_DONE)
        {
            size_t node_id = 0;
            TaggedNode* tn = nullptr;

            switch (state)
            {
            // Process the nodes in ready queue
            case PROCESS_READY_NODE:
            {
                if (ready.empty())
                {
                    state = (fuse_ready.empty() && elem_ready.empty()) ? WORK_DONE
                                                                       : PROCESS_FUSIABLE_NODE;
                    break;
                }

                node_id = ready.front();
                ready.pop();
                tn = &m_nodes[node_id];
                break;
            }
            // Process the nodes in fuse_ready queue
            case PROCESS_FUSIABLE_NODE:
            {
                if (fuse_ready.empty())
                {
                    if (elem_ready.empty())
                    {
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
                    }
                    else
                    {
                        state = PROCESS_ELEM_NODE;
                    }
                    break;
                }
                node_id = fuse_ready.front();
                fuse_ready.pop_front();
                tn = &m_nodes[node_id];

                if (!cur_group)
                    cur_group = std::make_shared<FuseGroup>();
                cur_group->nodes.push_back(node_id);
                break;
            }

            // Process the nodes in elem_ready queue
            case PROCESS_ELEM_NODE:
            {
                if (elem_ready.empty())
                {
                    if (cur_elemgroup && cur_elemgroup->nodes.size() > 0)
                    {
                        // append cur_elemgroup to cur_group
                        if (!cur_group)
                        {
                            cur_group = std::make_shared<FuseGroup>();
                        }
                        TaggedNode tn;
                        tn.elem_group = cur_elemgroup;
                        int new_id = m_nodes.size();
                        m_nodes.push_back(tn);
                        cur_group->nodes.push_back(new_id);
                        cur_elemgroup = nullptr;
                    }
                    state = PROCESS_FUSIABLE_NODE;
                    break;
                }
                node_id = elem_ready.front();
                elem_ready.pop_front();
                tn = &m_nodes[node_id];

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
                    auto& dst = m_nodes[edge->get_dst()->get_id()];
                    dst.ready_inputs++;

                    if (!dst.visited && dst.ready_inputs >= dst.node->get_in_edges().size())
                    {
                        AddNodeToReadyQueues(dst.node, ready, fuse_ready, elem_ready);
                    }
                }
            }
        } // while

        return groups;
    }

    void TagFusionGroupsOnGraph(std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> groups)
    {
        int next_fusion_group_id = 0;
        int next_elem_group_id = 0;

        for (auto group : *groups)
        {
            LOG_INFO << DebugStringFuseGroup(group);
            for (auto id : group->nodes)
            {
                enforce(id < m_nodes.size());
                TaggedNode& tn = m_nodes[id];
                if (id >= ELEM_GROUP_NODEID && tn.elem_group)
                {
                    for (auto elem_id : tn.elem_group->nodes)
                    {
                        enforce(elem_id < m_nodes.size());
                        TaggedNode& elem_tn = m_nodes[elem_id];
                        enforce_not_nullptr(elem_tn.node);

                        (*elem_tn.node)["elem_group_id"] = next_elem_group_id;
                        (*elem_tn.node)["fusion_group_id"] = next_fusion_group_id;
                    }
                    next_elem_group_id++;
                }
                else
                {
                    (*tn.node)["fusion_group_id"] = next_fusion_group_id;
                }
            }
            next_fusion_group_id++;
        }
    }

    std::string DebugStringFuseGroup(std::shared_ptr<FuseGroup> group)
    {
        std::ostringstream ret;
        ret << "========================Fusion Group =====================\n";

        auto PrintInfo = [this, &ret](const size_t id) {
            auto n = m_nodes[id];
            ret << id << " / " << n.node->get_id() << ":" << n.node->get_name() << "\t"
                << n.node->get_op_type() << "\n";
        };

        ret << "FUSING NODES: [\n";
        for (auto id : group->nodes)
        {
            if (id < ELEM_GROUP_NODEID)
            {
                PrintInfo(id);
            }
            else
            {
                ret << "((\n";
                for (auto eid : m_nodes[id].elem_group->nodes)
                {
                    PrintInfo(eid);
                }
                ret << ")) \n\n";
            }
        }
        ret << "]\n";
        return ret.str();
    }

private:
    std::shared_ptr<Graph> m_graph;
    std::vector<TaggedNode> m_nodes;
    size_t ELEM_GROUP_NODEID;

    std::unordered_set<std::string> op_blacklist;
    std::unordered_set<std::string> fusion_ops;
    std::unordered_map<std::string, int> host_inputs;
};

bool KernelFusionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    KernelFuseOptimizer optimizer(graph);
    return optimizer.Optimize();
}