// Microsoft (c) 2019, NNFusion Team

#include "assign_layout_pass.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool AssignLayoutPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    for (auto gnode : graph->get_nodes())
    {
        try
        {
            for (size_t i = 0; i < gnode->get_output_size(); ++i)
            {
                auto tv = gnode->get_outputs().at(i)->get_tensor_ptr();
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
            CHECK_FAIL_WITH_EXCEPTION(errors::InvalidArgument)
                << "Error with node " << gnode->get_unique_name() << ": " << e.what();
        }

        // TODO: to be removed once gnode input is used
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
            CHECK_FAIL_WITH_EXCEPTION(errors::InvalidArgument) << "Error with node " << *node
                                                               << ": " << e.what();
        }
        // end TODO
    }
    return true;
}
