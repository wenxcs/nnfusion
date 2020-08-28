// Microsoft (c) 2019, NNFusion Team
#include "dot_transpose_pass.hpp"
#include "kernel_selection.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"
#include "nnfusion/core/operators/op_define/dot.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

DEFINE_bool(fdot_transpose, false, "Dot transpose.");

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

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
    std::set<std::shared_ptr<GNode>> const_nodes = {};
    std::set<std::shared_ptr<GNode>> down_streams = {};

    // Find nodes with all constant upstream nodes
    for (auto& it : nodes)
    {
        if (it->get_op_type() == "Dot")
        {
            auto dot = std::dynamic_pointer_cast<nnfusion::op::Dot>(it->get_op_ptr());
            NNFUSION_CHECK_NOT_NULLPTR(dot);
            // already transposed
            if (dot->get_transpose_B())
                continue;

            auto in_edge_0 = it->get_in_edge(0);
            NNFUSION_CHECK(in_edge_0);
            auto input_0_gnode = in_edge_0->get_src();
            auto input_0_index = in_edge_0->get_src_output();

            auto in_edge_1 = it->get_in_edge(1);
            NNFUSION_CHECK(in_edge_1);
            auto input_1_gnode = in_edge_1->get_src();
            auto input_1_index = in_edge_1->get_src_output();

            // multiple reference to input0
            if (input_0_gnode->get_output_users(input_0_index).size() > 1)
            {
                continue;
            }

            auto const_op =
                std::dynamic_pointer_cast<nnfusion::op::Constant>(input_1_gnode->get_op_ptr());
            // only handle const input1, ignore weight because optimizer might update these const
            if (!input_1_gnode->is_constant() || input_1_gnode->get_shape().size() != 2 ||
                const_op->is_weight())
            {
                continue;
            }

            shared_ptr<KernelContext> ctx(new KernelContext(it));
            std::string identifier = generate_identifier(ctx);
            if (identifier == "")
                continue;

            auto common_kernel = cache_manager->fetch(identifier, "");
            auto transpose_kernel = cache_manager->fetch(identifier, "trans_b");
            if (common_kernel == "" || transpose_kernel == "")
            {
                continue;
            }

            ///\todo placeholder, deserialize db profile column and compare kernel time
            if (false)
            {
                continue;
            }

            // insert transpose
            auto trans_gnode = nnfusion::graph::numpy_transpose(
                input_1_gnode, nnfusion::AxisVector(), input_1_index);
            graph->add_node(trans_gnode);
            graph->add_edge(input_1_gnode, input_1_index, trans_gnode, 0);
            // reconnect dot node
            graph->remove_edge(in_edge_1);
            graph->add_edge(trans_gnode, 0, it, 1);
            dot->get_transpose_B() = true;
        }
    }

    return true;
}
