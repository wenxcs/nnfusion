// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "assign_layout_pass.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "nnfusion/util/log.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::graph::pass;

bool AssignLayoutPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    for (auto gnode : graph->get_nodes())
    {
        auto node = gnode->get_op_ptr();
        try
        {
            for (size_t i = 0; i < node->get_output_size(); ++i)
            {
                auto tv = node->get_output_tensor_ptr(i);
                if (nullptr == tv->get_tensor_layout())
                {
                    auto layout =
                        std::make_shared<ngraph::descriptor::layout::DenseTensorLayout>(*tv);
                    tv->set_tensor_layout(layout);
                }
            }
        }
        catch (const std::exception& e)
        {
            std::stringstream ss;
            ss << "Error with node " << *node << ": ";
            ss << e.what();
            LOG_ERR << ss.str();
            // TODO: how to handle error
            throw std::invalid_argument(ss.str());
        }
    }
    return true;
}