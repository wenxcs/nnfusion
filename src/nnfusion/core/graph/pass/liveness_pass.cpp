// Microsoft (c) 2019, NNFusion Team

#pragma once

#include <unordered_set>

#include "liveness_pass.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"
#include "nnfusion/util/log.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::graph::pass;

bool LivenessPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    auto gnodes = graph->get_ordered_ops();
    std::unordered_set<ngraph::descriptor::Tensor*> persistent_tensors;
    std::unordered_set<ngraph::descriptor::Tensor*> output_tensors;
    std::unordered_set<ngraph::descriptor::Tensor*> constant_tensors;

    for (const auto gnode : graph->get_parameters())
    {
        auto node = gnode->get_op_ptr();
        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            ngraph::descriptor::Tensor& tensor = node->get_output_tensor(i);
            persistent_tensors.insert(&tensor);
        }
    }

    for (const auto gnode : graph->get_outputs())
    {
        auto node = gnode->get_op_ptr();

        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            ngraph::descriptor::Tensor& tensor = node->get_output_tensor(i);
            persistent_tensors.insert(&tensor);
            output_tensors.insert(&tensor);
        }
    }

    for (const auto gnode : gnodes)
    {
        auto node = gnode->get_op_ptr();

        if (auto constant_node = std::dynamic_pointer_cast<ngraph::op::Constant>(node))
        {
            for (size_t i = 0; i < constant_node->get_output_size(); ++i)
            {
                ngraph::descriptor::Tensor& tensor = constant_node->get_output_tensor(i);
                constant_tensors.insert(&tensor);
            }
        }
    }

    std::unordered_set<ngraph::descriptor::Tensor*> currently_live;
    for (auto it = gnodes.rbegin(); it != gnodes.rend(); it++)
    {
        auto gnode = *it;
        const std::shared_ptr<ngraph::Node>& node = gnode->get_op_ptr();
        node->liveness_new_list.clear();
        node->liveness_free_list.clear();
        std::unordered_set<ngraph::descriptor::Tensor*> input_tensor_decls;
        for (ngraph::descriptor::Input& input_decl : node->get_inputs())
        {
            ngraph::descriptor::Tensor& tensor = input_decl.get_tensor();
            if (persistent_tensors.find(&tensor) == persistent_tensors.end())
            {
                input_tensor_decls.insert(&tensor);
            }
        }

        std::unordered_set<ngraph::descriptor::Tensor*> output_tensor_decls;
        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            ngraph::descriptor::Tensor& tensor = node->get_output_tensor(i);
            if (persistent_tensors.find(&tensor) == persistent_tensors.end())
            {
                output_tensor_decls.insert(&tensor);
            }
        }

        std::unordered_set<ngraph::descriptor::Tensor*> free_tensor_decls;
        std::unordered_set<ngraph::descriptor::Tensor*> new_tensor_decls;
        std::unordered_set<ngraph::descriptor::Tensor*> all_tensor_decls = input_tensor_decls;
        all_tensor_decls.insert(output_tensor_decls.begin(), output_tensor_decls.end());

        for (ngraph::descriptor::Tensor* tensor_decl : all_tensor_decls)
        {
            if (currently_live.find(tensor_decl) == currently_live.end())
            {
                // this is the last node that value is seen in
                // delete it at the end of the op
                currently_live.insert(tensor_decl);
                if (output_tensors.find(tensor_decl) == output_tensors.end() &&
                    constant_tensors.find(tensor_decl) == constant_tensors.end())
                {
                    // Don't free output tensors and constant tensors
                    free_tensor_decls.insert(tensor_decl);
                }
            }
        }

        for (ngraph::descriptor::Tensor* output_decl : output_tensor_decls)
        {
            auto currently_live_it = currently_live.find(output_decl);
            if (currently_live_it != currently_live.end())
            {
                new_tensor_decls.insert(output_decl);
                currently_live.erase(currently_live_it);
            }
        }
        node->liveness_free_list = free_tensor_decls;
        node->liveness_new_list = new_tensor_decls;
    }

    return true;
}