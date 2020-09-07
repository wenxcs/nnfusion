// Microsoft (c) 2019, NNFusion Team

#include "superscaler_dataparallelism_pass.hpp"
#include <unistd.h>
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/allreduce.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/core/operators/op_define/result.hpp"
using namespace nnfusion::graph;
using namespace nnfusion::op;
using namespace nnfusion::pass::graph;
using namespace std;
DEFINE_bool(fadd_sc_allreduce, false, "Add Allreduce operater after ApplyGradient operator.");
DEFINE_bool(fadd_sc_allreduce_fusion,
            false,
            "Add fused sc Allreduce operater after ApplyGradient operator.");
DEFINE_int32(sc_allreduce_fusion_num, -1, "set the number of adjacent allreduce_op to fuse.");
DEFINE_int32(sc_allreduce_fusion_size,
             -1,
             "set the floats of data to fuse: 67108864 is recommended.");
DEFINE_int32(sc_allreduce_fusion_time, -1, "set the timeout to fuse: 1000 millisecond by default.");

#define SC_ALLREDUCE_DEBUG
int SuperScalerDataParallelismPass::get_gradient_from_apply(std::shared_ptr<GNode> apply_node)
{
    //TODO: adapt for more apply op
    int weight_index = apply_node->get_in_edge(0)->get_src()->get_op_type() == "Variable" ? 0 : 1;
    return (weight_index + 1) % 2;
}

bool SuperScalerDataParallelismPass::is_apply_node(std::shared_ptr<GNode> apply_node)
{
    // skip nodes whose type are not Apply*.
    // TODO: now we only support "ApplyGradientDescent" op, we should add more Apply* in the future.
    if (apply_node->get_op_type() != "ApplyGradientDescent")
        return false; //TODO: adapt for more apply op
    NNFUSION_CHECK(apply_node->get_in_edges().size() == 2)
        << "ApplyGradientDescent node's in_edges :" << apply_node->get_in_edges().size();
    return true;
}

std::vector<std::vector<int>> SuperScalerDataParallelismPass::group_gradient_apply(
    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>> hash_to_gradient_apply)
{
    std::vector<std::vector<int>> gradient_key_subgroups;
    if (!sc_allreduce_fusion_enable)
    {
        for (int i = 0; i < hash_to_gradient_apply.size(); i++)
        {
            std::vector<int> subgroup;
            subgroup.push_back(i);
            gradient_key_subgroups.push_back(subgroup);
        }
    }
    else
    {
        int sc_allreduce_fusion_num = FLAGS_sc_allreduce_fusion_num;
        int sc_allreduce_fusion_size = FLAGS_sc_allreduce_fusion_size;
        int sc_allreduce_fusion_time = FLAGS_sc_allreduce_fusion_time;
        NNFUSION_LOG(INFO) << "[sc pass] sc_allreduce_fusion_num:" << sc_allreduce_fusion_num;
        NNFUSION_LOG(INFO) << "[sc pass] sc_allreduce_fusion_size:" << sc_allreduce_fusion_size;
        NNFUSION_LOG(INFO) << "[sc pass] sc_allreduce_fusion_time:" << sc_allreduce_fusion_time;
        if (sc_allreduce_fusion_num <= 0 && sc_allreduce_fusion_size <= 0 &&
            sc_allreduce_fusion_time <= 0)
        {
            sc_allreduce_fusion_num = hash_to_gradient_apply.size(); //concat all allreduce into one
            NNFUSION_LOG(INFO) << "[sc pass] reset sc_allreduce_fusion_num:"
                               << sc_allreduce_fusion_num;
        }
        std::vector<int> subgroup;
        std::vector<int> fused_sizes;
        if (sc_allreduce_fusion_num > 0)
        {
            int curr_fuse_size = 0;
            for (int i = 0; i < hash_to_gradient_apply.size(); i++)
            {
                int curr = shape_size(hash_to_gradient_apply[i].first->get_shape());
                curr_fuse_size += curr;
                //allreduce nodes are adjacent and sorted from back to front when backward by default
                subgroup.push_back(i);
                if (subgroup.size() >= sc_allreduce_fusion_num)
                {
                    gradient_key_subgroups.push_back(subgroup);
                    fused_sizes.push_back(curr_fuse_size);
                    subgroup.clear();
                    curr_fuse_size = 0;
                }
            }
            if (subgroup.size() != 0) // fuse the remaining allreduce nodes
            {
                gradient_key_subgroups.push_back(subgroup);
                fused_sizes.push_back(curr_fuse_size);
                subgroup.clear();
                curr_fuse_size = 0;
            }
        }
        else
        {
            // timeout and buffer_size
            NNFUSION_CHECK(sc_allreduce_fusion_time == -1)
                << "now sc_allreduce_fusion_time is not supported.";
            int curr_fuse_size = 0;
            for (int i = 0; i < hash_to_gradient_apply.size(); i++)
            {
                //TODO: timeout mechanism
                int curr = shape_size(hash_to_gradient_apply[i].first->get_shape());
                if (curr_fuse_size + curr > sc_allreduce_fusion_size)
                {
                    gradient_key_subgroups.push_back(subgroup);
                    fused_sizes.push_back(curr_fuse_size);
                    subgroup.clear();
                    curr_fuse_size = 0;
                }
                subgroup.push_back(i);
                curr_fuse_size += curr;
            }
            if (subgroup.size() != 0) // fuse the remaining allreduce nodes
            {
                gradient_key_subgroups.push_back(subgroup);
                fused_sizes.push_back(curr_fuse_size);
                subgroup.clear();
                curr_fuse_size = 0;
            }
        }
#ifdef SC_ALLREDUCE_DEBUG
        for (int i = 0; i < gradient_key_subgroups.size(); i++)
        {
            auto subgroup = gradient_key_subgroups[i];
            int fused_size = fused_sizes[i];
            NNFUSION_LOG(INFO) << "[sc pass] fused " << fused_size
                               << " bytes including gradient_id:";
            for (auto i : subgroup)
            {
                std::cout << i << ",";
            }
            std::cout << std::endl;
        }
#endif
    }
    return gradient_key_subgroups;
}

std::shared_ptr<GNode> SuperScalerDataParallelismPass::concat_into_one(
    std::shared_ptr<Graph>& graph,
    std::vector<int> subgroup,
    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>> hash_to_gradient_apply)
{
    std::shared_ptr<GNode> concat_node;
    if (!sc_allreduce_fusion_enable)
    {
        //each subgroup contains one gradient key
        NNFUSION_CHECK(subgroup.size() == 1)
            << "enable fused_allreduce to support fuse subgroup containing " << subgroup.size()
            << " gradients";
        auto iter = hash_to_gradient_apply.find(subgroup[0]);
        NNFUSION_CHECK(iter != hash_to_gradient_apply.end());
        concat_node = iter->second.first;
    }
    else
    {
        //TODO each subgroup contains some gradients
        //gradient->reshape->concat
        GNodeIndexVector concat_inputs;
#ifdef SC_ALLREDUCE_DEBUG
        NNFUSION_LOG(INFO) << "[sc pass] concat_into_one";
#endif
        for (int i : subgroup)
        {
            auto gradient_apply = hash_to_gradient_apply[i];
            auto gradient_node = gradient_apply.first;
#ifdef SC_ALLREDUCE_DEBUG
            std::cout << i << ": " << gradient_node->get_shape() << ",";
#endif
            int n = 0;
            nnfusion::AxisVector order = nnfusion::AxisVector(gradient_node->get_shape().size());
            std::generate(order.begin(), order.end(), [&n]() { return n++; });
            auto apply_node = gradient_apply.second;
            auto reshape_op = std::make_shared<op::Reshape>(
                order,
                nnfusion::Shape(1, shape_size(gradient_node->get_shape()))); //AxisVector={0, 1..}
            add_inplace(reshape_op, 0, 0, false);
            auto reshape_node =
                graph->add_node_and_edge(reshape_op, {gradient_node}); //output_index=0
            concat_inputs.push_back(GNodeIndex(reshape_node, 0));
        }
        auto concat_op = std::make_shared<nnfusion::op::Concat>(0);
        auto first_gradient_node = hash_to_gradient_apply[subgroup[0]].first;
        concat_op->set_name(first_gradient_node->get_name() + "_fusion_concat_node");
        concat_node = graph->add_node_and_edge(concat_op, {concat_inputs});
#ifdef SC_ALLREDUCE_DEBUG
        std::cout << std::endl << "concat:" << concat_node->get_shape() << std::endl;
#endif
    }
    return concat_node;
}

std::vector<std::pair<std::shared_ptr<GNode>, int>> SuperScalerDataParallelismPass::split_from_one(
    std::shared_ptr<Graph>& graph,
    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>> hash_to_gradient_apply,
    std::shared_ptr<GNode> allreduce_node,
    std::vector<int> subgroup)
{
    std::vector<std::pair<std::shared_ptr<GNode>, int>> allreduced_gradients;
    if (!sc_allreduce_fusion_enable)
    {
        NNFUSION_CHECK(subgroup.size() == 1);
        //each allreduce_node produce only one gradient
        allreduced_gradients.push_back(
            std::pair<std::shared_ptr<GNode>, int>(allreduce_node, subgroup[0]));
        return allreduced_gradients;
    }
    else
    {
        size_t cursor = 0;
        std::vector<size_t> lower{0};
        std::vector<size_t> upper{0};
        size_t allreduced_tensor_size = shape_size(allreduce_node->get_shape());
#ifdef SC_ALLREDUCE_DEBUG
        NNFUSION_LOG(INFO) << "[sc pass] slice_from_one ";
#endif
        for (int i : subgroup)
        {
            auto gradient_apply = hash_to_gradient_apply[i];
            auto gradient_node = gradient_apply.first;
            //allreduce->slice
            nnfusion::Shape gradient_shape =
                gradient_node->get_shape(); //default get_output_shape(output_index=0)
            cursor += shape_size(gradient_shape);
            upper[0] = cursor;
            NNFUSION_CHECK(cursor <= allreduced_tensor_size) << "slice range is out of buffer";
            auto slice_op = std::make_shared<nnfusion::op::Slice>(lower, upper);
            lower[0] = cursor;
            slice_op->set_name(gradient_node->get_name() + "_fusion_slice_node");
            auto slice_node = graph->add_node_and_edge(slice_op, {allreduce_node});
#ifdef SC_ALLREDUCE_DEBUG
            std::cout << i << ":" << slice_node->get_shape() << "=>";
#endif
            //allreduce->slice->reshape
            auto reshape_op = std::make_shared<op::Reshape>(nnfusion::AxisVector{0},
                                                            gradient_shape); //AxisVector={0, 1..}
            add_inplace(reshape_op, 0, 0, false);
            auto reshape_node = graph->add_node_and_edge(reshape_op, {slice_node}); //output_index=0
#ifdef SC_ALLREDUCE_DEBUG
            std::cout << reshape_node->get_shape() << ", ";
#endif
            allreduced_gradients.push_back(std::pair<std::shared_ptr<GNode>, int>(reshape_node, i));
        }
#ifdef SC_ALLREDUCE_DEBUG
        std::cout << std::endl;
#endif
    }
    return allreduced_gradients;
}

bool SuperScalerDataParallelismPass::add_allreduce(
    std::shared_ptr<Graph>& graph,
    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>> hash_to_gradient_apply)
{
    // Weight(weight_node) ----|
    //                         |
    //                         V
    // (gradient)     ApplyGradient-> Result
    std::vector<std::vector<int>> gradient_key_subgroups =
        group_gradient_apply(hash_to_gradient_apply);
    for (std::vector<int> subgroup : gradient_key_subgroups)
    {
        auto concat_node = concat_into_one(graph, subgroup, hash_to_gradient_apply);
#ifdef SC_ALLREDUCE_DEBUG
        NNFUSION_LOG(INFO) << "[sc pass] concat_node name:" << concat_node->get_name();
#endif
        // Weight(weight_node) ----------------|
        //                                     |
        //                                     |
        // (gradient)->reshape---|             |
        //                       V             V
        //(gradient)->reshape-> concat     ApplyGradient-> Result
        std::shared_ptr<GNode> allreduce_node;
        if (sc_allreduce_enable)
        {
            auto allreduce_op = std::make_shared<AllReduce>();
            allreduce_node = graph->add_node_and_edge(allreduce_op, {concat_node});
        }
        NNFUSION_LOG(INFO) << "[sc pass] allreduce name:" << allreduce_node->get_name();
        // Weight(weight_node) ----------------------------|
        //                                                 |
        //                                                 |
        // (gradient)->reshape---|                         |
        //                       V                         V
        //(gradient)->reshape-> concat --> allreduce     ApplyGradient-> Result
        std::vector<std::pair<std::shared_ptr<GNode>, int>> reshape_nodes =
            split_from_one(graph, hash_to_gradient_apply, allreduce_node, subgroup);
#ifdef SC_ALLREDUCE_DEBUG
        std::cout << "[sc pass] ";
        for (std::pair<std::shared_ptr<GNode>, int> reshape_idx : reshape_nodes)
        {
            std::cout << "reshape name:" << reshape_idx.first->get_name() << ", ";
        }
        std::cout << std::endl;
#endif
        // Weight(weight_node) --------------------------------------------------------------------|
        //                                                                                         |
        //                                                                                         |
        // (gradient)->reshape---|                                                                 V
        //                       |                                              |->reshape->ApplyGradient-> Result
        //                       V                                              |
        //(gradient)->reshape-> concat(concated_gradient) --> allreduce --> slice->reshape->ApplyGradient-> Result
        for (std::pair<std::shared_ptr<GNode>, int> reshape_idx : reshape_nodes)
        {
            std::shared_ptr<GNode> apply_node = hash_to_gradient_apply[reshape_idx.second].second;
            int gradient_index = get_gradient_from_apply(apply_node);
            graph->add_edge(reshape_idx.first, 0, apply_node, gradient_index);
        }
    }
    return true;
}

bool SuperScalerDataParallelismPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    sc_allreduce_enable = FLAGS_fadd_sc_allreduce;
    sc_allreduce_fusion_enable = FLAGS_fadd_sc_allreduce_fusion;
    if (!sc_allreduce_enable)
        return true;
    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>> hash_to_gradient_apply;
    // group gradient and apply* op from n-th layer to 1st layer
    for (int i = graph->get_outputs().size() - 1; i >= 0; i--)
    {
        auto result_node = graph->get_outputs()[i];
#if 0
        std::cout<< "before superscaler: " << result_node->get_name() << std::endl;
        for(auto edge: result_node->get_in_edges())
        {
            std::cout<< "in: " << edge->get_src()->get_name() << std::endl;
            for(auto e: edge->get_src()->get_in_edges())
            {
                std::cout << e->get_src()->get_name() << std::endl;
            }
        }
#endif
        // the apply node followed by result node. so check result_node's input node
        if (!is_apply_node(result_node->get_in_edge(0)->get_src()))
            continue;
        NNFUSION_CHECK(result_node->get_in_edges().size() == 1)
            << "result node has other input except apply op:";
        auto apply_node = result_node->get_in_edge(0)->get_src();
        // Weight(weight_node) ----|
        //                         |
        //                         V
        // (gradient) --> ApplyGradient-> Result
        // find the gradient out.
        int gradient_index = get_gradient_from_apply(apply_node);
        std::shared_ptr<GNode> gradient_node = apply_node->get_in_edge(gradient_index)->get_src();
        graph->remove_edge(apply_node->get_in_edge(gradient_index));
        // Weight(weight_node) ----|
        //                         |
        //                         V
        // (gradient)     ApplyGradient-> Result
        hash_to_gradient_apply[hash_to_gradient_apply.size()] =
            std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>(gradient_node, apply_node);
#ifdef SC_ALLREDUCE_DEBUG
        NNFUSION_LOG(INFO) << "[sc pass] " << gradient_node->get_name()
                           << " hash id:" << hash_to_gradient_apply.size() - 1;
#endif
    }
    bool is_success = add_allreduce(graph, hash_to_gradient_apply);
#if 0
    for(auto result_node: graph->get_outputs())
    {
        std::cout<< "after superscaler: " << result_node->get_name() << std::endl;
        for(auto edge: result_node->get_in_edges())
        {
            auto apply_node = edge->get_src();
            std::cout<< "in: " apply_node<< ->get_name() << std::endl;//in apply node
            for(auto e: apply_node->get_in_edges())
            {
                std::cout << e->get_src()->get_name() << std::endl;//reshape or allreduce
            }
        }
    }
#endif
    return is_success;
}
