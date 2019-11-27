// Microsoft (c) 2019, NNFusion Team

#include "kernel_fusion_pass.hpp"
#include <queue>
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"

#include "gflags/gflags.h"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace ngraph::op::util;

DEFINE_int32(fkernel_fusion_level, 2, "");

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

    size_t output_size = 0; // used in element_group

    std::vector<size_t> nodes;
};

class KernelFuseOptimizer
{
public:
    KernelFuseOptimizer(std::shared_ptr<Graph> g)
        : m_graph(g)
    {
        m_nodes.resize(m_graph->get_max_node_id());
        ELEM_GROUP_NODEID = m_nodes.size();
        RegisterFusionOps();
    }

    bool Optimize()
    {
        int fusion_level = FLAGS_fkernel_fusion_level;
        if (fusion_level > 0)
        {
            std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> fuse_groups =
                ExtractFusionGroups();
            if (fuse_groups != nullptr && fuse_groups->size() > 0)
            {
                if (fusion_level > 1)
                {
                    FuseReshapeAndBroadcast(fuse_groups);
                }
                TagFusionGroupsOnGraph(fuse_groups);
                return true;
            }
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
            CHECK(id < m_nodes.size());
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
                    state = (fuse_ready.empty() && elem_ready.empty()) ? WORK_DONE
                                                                       : PROCESS_FUSIABLE_NODE;
                    break;
                }

                node_id = ready.front();
                ready.pop();
                tn = m_nodes[node_id];
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
                tn = m_nodes[node_id];

                if (!cur_group)
                    cur_group = std::make_shared<FuseGroup>();
                cur_group->nodes.push_back(node_id);
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
                    state = PROCESS_FUSIABLE_NODE;
                    break;
                }

                node_id = elem_ready.front();
                elem_ready.pop_front();
                tn = m_nodes[node_id];

                size_t tensor_size = shape_size(tn->node->get_outputs().at(0)->get_shape());
                if (cur_elemgroup && cur_elemgroup->output_size != tensor_size)
                {
                    AppendElementGroup();
                }

                if (!cur_elemgroup)
                {
                    cur_elemgroup = std::make_shared<FuseGroup>();
                }
                cur_elemgroup->nodes.push_back(node_id);
                cur_elemgroup->output_size = tensor_size;
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
                    dst->ready_inputs++;

                    if (!(dst->visited) && (dst->ready_inputs >= dst->node->get_in_edges().size()))
                    {
                        AddNodeToReadyQueues(dst->node, ready, fuse_ready, elem_ready);
                    }
                }
            }
        } // while

        return groups;
    }

    void FuseReshapeAndBroadcast(std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> groups)
    {
        int next_fusion_group_id = 0;
        int next_elem_group_id = 0;

        for (auto group : *groups)
        {
            for (auto id : group->nodes)
            {
                CHECK(id < m_nodes.size());
                auto tn = m_nodes[id];
                if (id >= ELEM_GROUP_NODEID && tn->elem_group)
                {
                    std::vector<size_t> fusable_input_nodes;
                    for (auto elem_id : tn->elem_group->nodes)
                    {
                        CHECK(elem_id < m_nodes.size());
                        auto elem_tn = m_nodes[elem_id];
                        CHECK_NOT_NULLPTR(elem_tn->node);
                        for (auto in_edge : elem_tn->node->get_in_edges())
                        {
                            if (in_edge->is_control_edge())
                                continue;
                            auto input_node = in_edge->get_src();
                            while (input_node)
                            {
                                auto op = input_node->get_op_ptr();
                                if (auto bc = std::dynamic_pointer_cast<ngraph::op::Broadcast>(op))

                                {
                                    if (bc->is_inner_broadcast() || bc->is_outer_broadcast())
                                    {
                                        fusable_input_nodes.push_back(input_node->get_id());
                                        // cannot fuse more nodes before broadcast
                                        break;
                                    }
                                }
                                else if (auto rs =
                                             std::dynamic_pointer_cast<ngraph::op::Reshape>(op))
                                {
                                    if (!(rs->get_is_transpose()) &&
                                        (shape_size(input_node->get_outputs().at(0)->get_shape()) ==
                                         shape_size(
                                             elem_tn->node->get_outputs().at(0)->get_shape())))
                                        fusable_input_nodes.push_back(input_node->get_id());
                                }
                                else
                                {
                                    break;
                                }

                                bool input_set = false;
                                for (auto edge : input_node->get_in_edges())
                                {
                                    if (!edge->is_control_edge())
                                    {
                                        CHECK(input_set == false)
                                            << "Reshape and Broadcast can only have 1 input!";
                                        input_node = edge->get_src();
                                        input_set = true;
                                    }
                                }
                            }
                        }
                    }
                    // insert reshape and broadcast nodes into this element group
                    tn->elem_group->nodes.insert(tn->elem_group->nodes.begin(),
                                                 fusable_input_nodes.begin(),
                                                 fusable_input_nodes.end());
                }
            }
        }
    }

    void TagFusionGroupsOnGraph(std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> groups)
    {
        int next_fusion_group_id = 0;
        int next_elem_group_id = 0;

        for (auto group : *groups)
        {
            // LOG(INFO) << DebugStringFuseGroup(group);
            for (auto id : group->nodes)
            {
                CHECK(id < m_nodes.size());
                auto tn = m_nodes[id];
                if (id >= ELEM_GROUP_NODEID && tn->elem_group)
                {
                    for (auto elem_id : tn->elem_group->nodes)
                    {
                        CHECK(elem_id < m_nodes.size());
                        auto elem_tn = m_nodes[elem_id];
                        CHECK_NOT_NULLPTR(elem_tn->node);

                        (*(elem_tn->node))["elem_group_id"] = next_elem_group_id;
                        (*(elem_tn->node))["fusion_group_id"] = next_fusion_group_id;
                    }
                    next_elem_group_id++;
                }
                else
                {
                    (*(tn->node))["fusion_group_id"] = next_fusion_group_id;
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
            ret << id << " / " << n->node->get_id() << ":" << n->node->get_name() << "\t"
                << n->node->get_op_type() << "\n";
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
                for (auto eid : m_nodes[id]->elem_group->nodes)
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
    std::vector<std::shared_ptr<TaggedNode>> m_nodes;
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