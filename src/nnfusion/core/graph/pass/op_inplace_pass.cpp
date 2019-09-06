// Microsoft (c) 2019, NNFusion Team


#include "op_inplace_pass.hpp"
#include "../gnode.hpp"
#include "../graph.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::graph::pass;

bool OpInplacePass::run_on_graph(std::shared_ptr<Graph>& graph)
{    
    for (auto node : graph->get_nodes())
    {   
        // add inplace tag for reshape op if !op->get_is_transpose() || op element num < 2
        if (node->get_op_type() == "Reshape")
        {
            std::shared_ptr<ngraph::op::Reshape> reshape =
                std::static_pointer_cast<ngraph::op::Reshape>(node->get_op_ptr());

            ngraph::Shape result_shape = reshape->get_output_shape();
            size_t result_shape_product = ngraph::shape_size(result_shape);

            if (!reshape->get_is_transpose() || result_shape_product < 2)
            {
                auto op_annotations = reshape->get_op_annotations();
                if (op_annotations)
                {
                    // pass-through
                    op_annotations->add_in_place_oi_pair({0, 0, false});
                }
                else
                {
                    op_annotations = std::make_shared<ngraph::op::util::OpAnnotations>();
                    // pass-through
                    op_annotations->add_in_place_oi_pair({0, 0, false});
                    reshape->set_op_annotations(op_annotations);
                }
            }
        }

        if (node->get_op_type() == "Result")
        {
            std::shared_ptr<ngraph::op::Result> result =
                std::static_pointer_cast<ngraph::op::Result>(node->get_op_ptr());

            auto op_annotations = result->get_op_annotations();
            if (op_annotations)
                {
                    // pass-through
                    op_annotations->add_in_place_oi_pair({0, 0, false});
                }
                else
                {
                    op_annotations = std::make_shared<ngraph::op::util::OpAnnotations>();
                    // pass-through
                    op_annotations->add_in_place_oi_pair({0, 0, false});
                    result->set_op_annotations(op_annotations);
                }               
        }
    }
    return true;
}