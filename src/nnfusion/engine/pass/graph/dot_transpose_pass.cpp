// Microsoft (c) 2019, NNFusion Team
#include "dot_transpose_pass.hpp"
#include "kernel_selection.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"
#include "nnfusion/core/operators/op_define/dot.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

DEFINE_bool(fdot_transpose, false, "Dot transpose.");

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

namespace
{
    float fetch_kernel_time(
        const string& identifier,
        const set<string>& tags, // should exactly match every tag, no more no less
        std::shared_ptr<nnfusion::cache::KernelCacheManager> cache_manager,
        NNFusion_DeviceType device)
    {
        float kernel_time = 0;
        nnfusion::cache::kernel kernel_instance =
            cache_manager->fetch_with_tags(identifier, "CUDA", tags);
        string device_str = get_device_str(device);
        if (kernel_instance.profile.find(device_str) != kernel_instance.profile.end())
        {
            kernel_time = kernel_instance.profile.at(device_str);
        }
        return kernel_time;
    }
}

bool DotTransposePass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    bool using_pass = FLAGS_fdot_transpose;
    if (!using_pass)
        return true;

    auto cache_manager = std::make_shared<cache::KernelCacheManager>();
    if (!cache_manager->is_valid())
    {
        NNFUSION_LOG(INFO) << "No valid kernel cache, ignore dot transpose pass";
        return true;
    }

    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();

    // Find nodes with all constant upstream nodes
    for (auto& it : nodes)
    {
        if (it->get_op_type() != "Dot")
        {
            continue;
        }
        // kernel already selected
        if ((*it)["Kernel_Selection_Result"].is_valid())
        {
            continue;
        }
        if (!(*it)["DeviceType"].is_valid())
        {
            NNFUSION_LOG(NNFUSION_WARNING)
                << "GNode DeviceType should be assigned before this pass：" << it->get_name();
            continue;
        }

        auto n_device_type = (*it)["DeviceType"].as<NNFusion_DeviceType>();
        auto dot = std::dynamic_pointer_cast<nnfusion::op::Dot>(it->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(dot);
        // already transposed
        if (dot->get_transpose_B())
        {
            continue;
        }
        auto in_edge_1 = it->get_in_edge(1);
        NNFUSION_CHECK(in_edge_1);
        auto input_1_gnode = in_edge_1->get_src();
        auto input_1_index = in_edge_1->get_src_output();
        // input1 should be a const and only referenced by current dot
        if (!input_1_gnode->is_constant() || input_1_gnode->get_shape().size() != 2 ||
            input_1_gnode->get_output_users(input_1_index).size() > 1)
        {
            continue;
        }
        auto const_op =
            std::dynamic_pointer_cast<nnfusion::op::Constant>(input_1_gnode->get_op_ptr());
        // ignore weight because optimizer might inplace update them
        if (const_op->is_weight())
        {
            continue;
        }

        shared_ptr<KernelContext> ctx(new KernelContext(it));
        std::string identifier = generate_identifier(ctx);
        if (identifier == "")
        {
            continue;
        }
        float dot_time = fetch_kernel_time(identifier, set<string>{}, cache_manager, n_device_type);
        float transpose_dot_time =
            fetch_kernel_time(identifier, set<string>{"transB"}, cache_manager, n_device_type);
        if (dot_time == 0 || transpose_dot_time == 0 || dot_time <= transpose_dot_time)
        {
            continue;
        }

        // insert transpose
        auto trans_gnode =
            nnfusion::graph::numpy_transpose(input_1_gnode, nnfusion::AxisVector(), input_1_index);
        graph->add_node(trans_gnode);
        graph->add_edge(input_1_gnode, input_1_index, trans_gnode, 0);
        // reconnect dot node
        graph->remove_edge(in_edge_1);
        graph->add_edge(trans_gnode, 0, it, 1);
        dot->get_transpose_B() = true;
    }

    return true;
}
