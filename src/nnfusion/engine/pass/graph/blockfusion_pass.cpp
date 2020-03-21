// Microsoft (c) 2019, NNFusion Team

#include "blockfusion_pass.hpp"
#include <queue>
#include "blockfusion/blockfusion.hpp"
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/cuda_gpu/kernels/blockfusion_fused.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion;
using namespace nnfusion::blockfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

DEFINE_int32(fblockfusion_level, 0, "");
DECLARE_string(fdefault_device);

const static size_t DEFAULT_GROUP_ID = -1;

struct FusionGroup
{
    FusionGroup(size_t g_id = DEFAULT_GROUP_ID)
        : id(g_id)
    {
    }
    size_t id;
    std::vector<size_t> nodes;
    std::vector<size_t> sub_group;
    std::vector<std::shared_ptr<KernelEmitter>> block_kernels;
};

struct TaggedNode
{
    TaggedNode()
        : node(nullptr)
        , group_id(DEFAULT_GROUP_ID)
        , ready_inputs(0)
        , visited(false)
    {
    }

    std::shared_ptr<GNode> node;
    size_t group_id;
    size_t ready_inputs;
    bool visited;
};

class BlockFuseOptimizer
{
public:
    BlockFuseOptimizer(std::shared_ptr<Graph> g)
        : m_graph(g)
    {
        m_nodes.resize(m_graph->get_max_node_id());
    }

    bool Optimize()
    {
        int fusion_level = FLAGS_fblockfusion_level;
        if (fusion_level > 0)
        {
            std::shared_ptr<std::vector<std::shared_ptr<FusionGroup>>> fuse_groups =
                ExtractFusionGroups();
            for (auto group : *fuse_groups)
            {
                // Todo: The group split policy is to be simplified
                switch (FuseGroupOnGraph(group))
                {
                case 0: NNFUSION_LOG(INFO) << "Group Fused\n"; break;
                case -1: NNFUSION_LOG(INFO) << "Group Skept\n"; break;
                case 1:
                    auto nodes = group->nodes;
                    auto sub_group = group->sub_group;
                    auto block_kernels = group->block_kernels;
                    std::shared_ptr<FusionGroup> sub_group_left = std::make_shared<FusionGroup>();
                    std::shared_ptr<FusionGroup> sub_group_right = std::make_shared<FusionGroup>();
                    size_t group_index = sub_group.size() / 2;
                    size_t node_index =
                        (group_index == 0) ? nodes.size() / 2 : sub_group[group_index];
                    while (true)
                    {
                        NNFUSION_CHECK(node_index > 0 && node_index < nodes.size());
                        sub_group_left->nodes =
                            std::vector<size_t>(nodes.begin(), nodes.begin() + node_index);
                        sub_group_right->nodes =
                            std::vector<size_t>(nodes.begin() + node_index, nodes.end());
                        sub_group_left->block_kernels = std::vector<std::shared_ptr<KernelEmitter>>(
                            block_kernels.begin(), block_kernels.begin() + node_index);
                        sub_group_right->block_kernels =
                            std::vector<std::shared_ptr<KernelEmitter>>(
                                block_kernels.begin() + node_index, block_kernels.end());
                        if (FuseGroupOnGraph(sub_group_left) != 1)
                        {
                            if (FuseGroupOnGraph(sub_group_right) == 1)
                            {
                                nodes =
                                    std::vector<size_t>(nodes.begin() + node_index, nodes.end());
                                block_kernels = std::vector<std::shared_ptr<KernelEmitter>>(
                                    block_kernels.begin() + node_index, block_kernels.end());

                                sub_group = std::vector<size_t>(sub_group.begin() + group_index,
                                                                sub_group.end());
                                for (size_t i = 0; i < sub_group.size(); i++)
                                    sub_group[i] -= node_index;
                                group_index = sub_group.size() / 2;
                                node_index =
                                    (group_index == 0) ? nodes.size() / 2 : sub_group[group_index];
                            }
                            else
                            {
                                break;
                            }
                        }
                        else
                        {
                            if (group_index == 0)
                            {
                                node_index /= 2;
                            }
                            else
                            {
                                group_index /= 2;
                                node_index =
                                    (group_index == 0) ? node_index / 2 : sub_group[group_index];
                            }
                        }
                    }
                    break;
                }
            }
        }
        return true;
    }

private:
    void verify_node(size_t node_id,
                     std::shared_ptr<GNode> node,
                     std::shared_ptr<FusionGroup> cur_group)
    {
        NNFUSION_CHECK_NOT_NULLPTR(node);

        if (!(*node)["Kernel_Selection_Result"].is_valid())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "Kernel should be emitted before this pass:"
                                           << node->get_name();
            return;
        }
        auto emitted_kernel = (*node)["Kernel_Selection_Result"]
                                  .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
        KernelEmitter::Pointer kernel = nullptr;

        if (emitted_kernel.second->get_or_emit_source() == nullptr)
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "Kernel should be emitted before this pass:"
                                           << node->get_name();
        }
        else
        {
            kernel = emitted_kernel.second;
            if (std::dynamic_pointer_cast<BlockCudaEmitter>(kernel) == nullptr)
            {
                NNFUSION_LOG(INFO) << "Kernel skept in block fusion: " << node->get_name();
                return;
            }
            cur_group->nodes.push_back(node_id);
            cur_group->block_kernels.push_back(kernel);
        }
    }

    // currently only topological order supported
    std::shared_ptr<std::vector<std::shared_ptr<FusionGroup>>> ExtractFusionGroups()
    {
        size_t GROUP_ID = 0;
        std::shared_ptr<std::vector<std::shared_ptr<FusionGroup>>> groups =
            std::make_shared<std::vector<std::shared_ptr<FusionGroup>>>();
        std::queue<size_t> ready;
        std::vector<size_t> sub_group{0};
        std::vector<size_t> next_sub_group;

        size_t sub_gid = 0;
        for (auto node : m_graph->get_nodes())
        {
            size_t id = node->get_id();
            m_nodes[id] = std::make_shared<TaggedNode>();
            m_nodes[id]->node = node;
            if (!(m_nodes[id]->visited) &&
                (m_nodes[id]->ready_inputs == node->get_in_edges().size()))
            {
                ready.push(id);
                sub_group.push_back(++sub_gid);
            }
        }

        while (!ready.empty())
        {
            size_t n_topo = ready.size();
            NNFUSION_CHECK(sub_group.size() > 1 && n_topo == sub_group.back());

            sub_gid = 0;
            next_sub_group = std::vector<size_t>{0};
            std::shared_ptr<FusionGroup> cur_group = std::make_shared<FusionGroup>(GROUP_ID++);
            for (size_t i = 1; i < sub_group.size(); i++)
            {
                cur_group->sub_group.push_back(cur_group->nodes.size());
                for (size_t j = sub_group[i - 1]; j < sub_group[i]; j++)
                {
                    size_t node_id = ready.front();
                    ready.pop();
                    auto tn = m_nodes[node_id];
                    tn->visited = true;
                    tn->group_id = cur_group->id;

                    verify_node(node_id, tn->node, cur_group);

                    for (auto edge : tn->node->get_out_edges())
                    {
                        size_t dst_id = edge->get_dst()->get_id();
                        auto dst = m_nodes[dst_id];
                        dst->ready_inputs++;

                        NNFUSION_CHECK(!(dst->visited));
                        if (dst->ready_inputs >= dst->node->get_in_edges().size())
                        {
                            ready.push(dst_id);
                            sub_gid++;
                        }
                    }
                    if (sub_gid > next_sub_group.back())
                    {
                        next_sub_group.push_back(sub_gid);
                    }
                }
                if (cur_group->sub_group.back() == cur_group->nodes.size())
                {
                    cur_group->sub_group.pop_back();
                }
            }
            sub_group = next_sub_group;
            if (cur_group->nodes.size() > 0)
            {
                NNFUSION_CHECK(cur_group->sub_group.back() < cur_group->nodes.size());
                groups->push_back(cur_group);
            }
        }
        for (auto node : m_graph->get_nodes())
        {
            auto tn = m_nodes[node->get_id()];
            NNFUSION_CHECK(tn->visited);
            NNFUSION_CHECK(tn->group_id != DEFAULT_GROUP_ID);
        }
        return groups;
    }

    bool SkipGroupOnProfilingResult(blockfusion::ProfilingResult profiling_result)
    {
        NNFUSION_LOG(INFO) << profiling_result.get_debug_string();

        if (profiling_result.profile_device)
        {
            // skip group when there is only one kernel in this group
            if (profiling_result.num_kernels <= 1)
            {
                NNFUSION_LOG(INFO) << "BlockFusion: skip group, num_kernels <= 1";
                return true;
            }

            // skip group when there are too many large kernels in this group
            if (profiling_result.num_large_kernels >= profiling_result.num_kernels)
            {
                NNFUSION_LOG(INFO) << "BlockFusion: skip group, too many large kernels";
                return true;
            }

            // skip group when BlockFusion gets no gain
            if (profiling_result.fused_estimation_time >= profiling_result.normal_execution_time)
            {
                NNFUSION_LOG(INFO)
                    << "BlockFusion: skip group, fused_estimation_time >= normal_execution_time";
                return true;
            }
        }

        if (profiling_result.profile_codegen)
        {
            //
        }

        return false;
    }

    int FuseGroupOnGraph(const std::shared_ptr<FusionGroup> group)
    {
        //  NNFUSION_LOG(INFO) << DebugStringFuseGroup(group);
        // codegen for the block fusion node, 1024 stands for the max number of block per kernel
        auto virtual_device_p = std::make_shared<BlockParallelDevice>(1024);
        BlockParallelDevice& virtual_device = *virtual_device_p;
        for (auto kernel : group->block_kernels)
        {
            // Todo: integrate the interface of profiling result, 10 stands for 10us
            auto kernel_metric = std::make_shared<KernelMetric>();
            kernel_metric->duration = 10;
            virtual_device.schedule_kernel(std::dynamic_pointer_cast<BlockCudaEmitter>(kernel),
                                           kernel_metric);
        }

        auto blockfusion_profiler = BlockFusionProfiler();
        blockfusion_profiler.set_profiling_context(virtual_device_p, nullptr);
        if (SkipGroupOnProfilingResult(blockfusion_profiler.get_profiling_result()))
        {
            return -1;
        }

        auto code_generator_p = std::make_shared<BlockFusionCudaCodegen>(
            std::make_shared<KernelContext>(), virtual_device.get_block_executor_program());
        BlockFusionCudaCodegen& code_generator = *code_generator_p;
        auto blockfusion_func = code_generator.get_or_emit_source();
        auto kernel_reg = KernelRegistry::Global()->FindKernelRegistration(
            "BlockFusionFused", CUDA_GPU, DT_FLOAT);
        NNFUSION_CHECK_NOT_NULLPTR(kernel_reg);
        auto ctx = code_generator.get_kernel_context(); // for codegenerator
        auto kernel = kernel_reg->m_factory(ctx);
        auto blockfusion_fused_kernel = std::dynamic_pointer_cast<BlockFusionFused>(kernel);
        NNFUSION_CHECK_NOT_NULLPTR(blockfusion_fused_kernel);
        blockfusion_fused_kernel->set_blockfusion_function(blockfusion_func);
        kernel->get_or_emit_source();

        blockfusion_profiler.set_profiling_context(virtual_device_p, code_generator_p);
        if (SkipGroupOnProfilingResult(blockfusion_profiler.get_profiling_result()))
        {
            return -1;
        }

        // auto ctx = std::make_shared<KernelContext>();
        // auto kernel = std::make_shared<BlockFusionCudaCodegen>(
        //     ctx, virtual_device.get_block_executor_program());
        // kernel->get_or_emit_source();

        //  NNFUSION_LOG(INFO) << virtual_device.DebugStringBE();

        // not necessary to be a NoOp
        auto fused_op = std::make_shared<nnfusion::op::NoOp>("blockfusion_kernel");
        GNodeVector empty_inputs;
        auto fused_node = std::make_shared<GNode>(fused_op, empty_inputs);
        ctx->gnode = fused_node;

        (*fused_node)["Kernel_Selection_Result"] = std::make_pair(CUDA_GPU, kernel);

        // rewrite the graph by replacing the group with fused node
        m_graph->add_node(fused_node);
        size_t next_input_id = 0;
        size_t next_output_id = 0;
        std::unordered_set<std::shared_ptr<GNode>> internal_nodes;

        for (auto node_id : group->nodes)
        {
            auto node = m_nodes[node_id]->node;
            for (const auto& in_edge : node->get_in_edges())
            {
                if (std::find(group->nodes.begin(),
                              group->nodes.end(),
                              in_edge->get_src()->get_id()) != group->nodes.end())
                {
                    continue;
                }
                auto input_id = in_edge->is_control_edge() ? Graph::kControlSlot : next_input_id++;
                if (input_id != Graph::kControlSlot)
                {
                    fused_node->set_input(input_id,
                                          node->get_inputs().at(in_edge->get_dst_input()));
                }
                m_graph->add_edge(
                    in_edge->get_src(), in_edge->get_src_output(), fused_node, input_id);
            }

            for (const auto& out_edge : node->get_out_edges())
            {
                if (std::find(group->nodes.begin(),
                              group->nodes.end(),
                              out_edge->get_dst()->get_id()) != group->nodes.end())
                {
                    continue;
                }
                auto output_id =
                    out_edge->is_control_edge() ? Graph::kControlSlot : next_output_id++;
                if (output_id != Graph::kControlSlot)
                {
                    fused_node->set_output(output_id,
                                           node->get_outputs().at(out_edge->get_src_output()));
                }
                m_graph->add_edge(
                    fused_node, output_id, out_edge->get_dst(), out_edge->get_dst_input());
            }
        }

        // ROCm can only support maximum 70 args for single kernel
        // CUDA support maxumum 4096 bytes for parameter space
        if (fused_node->get_in_edges().size() + fused_node->get_out_edges().size() >= 128)
        {
            m_graph->remove_node(fused_node);
            return 1;
        }
        else
        {
            for (auto node_id : group->nodes)
            {
                m_graph->remove_node(m_nodes[node_id]->node);
            }
        }
        return 0;
    }

    // schedule kernels onto parallel devices

    // Debug string for graph grouping
    std::string DebugStringFuseGroup(std::shared_ptr<FusionGroup> group)
    {
        std::ostringstream ret;
        ret << "========================Fusion Group =====================\n";

        auto PrintInfo = [this, &ret](const size_t id) {
            auto n = m_nodes[id];
            ret << id << " / " << n->node->get_id() << ":" << n->node->get_name() << "\t"
                << n->node->get_op_type() << "\n";
        };

        ret << "FUSING NODES: [\n";
        for (auto node_id : group->nodes)
        {
            ret << "((\n";
            PrintInfo(node_id);
            ret << ")) \n\n";
        }
        ret << "]\n";
        return ret.str();
    }

private:
    std::shared_ptr<Graph> m_graph;
    std::vector<std::shared_ptr<TaggedNode>> m_nodes;
};

bool BlockFusionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    auto dev_name = FLAGS_fdefault_device;
    if (dev_name == "ROCm" || dev_name == "CUDA")
    {
        NNFUSION_LOG(INFO) << "device: " << dev_name;
        BlockFuseOptimizer optimizer(graph);
        return optimizer.Optimize();
    }
    else
    {
        return false;
    }
}