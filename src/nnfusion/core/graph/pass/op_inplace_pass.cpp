// Microsoft (c) 2019, NNFusion Team

#include "op_inplace_pass.hpp"
#include "../gnode.hpp"
#include "../graph.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::graph::pass;

bool OpInplacePass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    for (auto node : graph->get_nodes())
    {
        if (auto op = std::dynamic_pointer_cast<ngraph::op::util::ArithmeticReduction>(
                node->get_op_ptr()))
        {
            ngraph::AxisSet reduce_axes = op->get_reduction_axes();
            auto input_shape_product = shape_size(op->get_input_shape(0));
            auto output_shape_product = shape_size(op->get_output_shape(0));
            if (reduce_axes.empty() || input_shape_product == output_shape_product)
            {
                AddInplace(op, 0, 0);
            }
        }

        // add inplace tag for reshape op if !op->get_is_transpose() || op element num < 2
        else if (auto op = std::dynamic_pointer_cast<ngraph::op::Reshape>(node->get_op_ptr()))
        {
            ngraph::Shape result_shape = op->get_output_shape();
            size_t result_shape_product = ngraph::shape_size(result_shape);

            if (!op->get_is_transpose() || result_shape_product < 2)
            {
                AddInplace(op, 0, 0);
            }
        }

        else if (auto op = std::dynamic_pointer_cast<ngraph::op::Result>(node->get_op_ptr()))
        {
            AddInplace(op, 0, 0);
        }

        else if (auto op = std::dynamic_pointer_cast<ngraph::op::Broadcast>(node->get_op_ptr()))
        {
            ngraph::AxisSet broadcast_axes = op->get_broadcast_axes();
            auto input_shape_product = shape_size(op->get_input_shape(0));
            auto output_shape_product = shape_size(op->get_output_shape(0));

            if (broadcast_axes.empty() || input_shape_product == output_shape_product)
            {
                AddInplace(op, 0, 0);
            }
        }

        else if (auto op = std::dynamic_pointer_cast<ngraph::op::Reduce>(node->get_op_ptr()))
        {
            ngraph::AxisSet reduce_axes = op->get_reduction_axes();
            auto input_shape_product = shape_size(op->get_input_shape(0));
            auto output_shape_product = shape_size(op->get_output_shape(0));
            if (reduce_axes.empty() || input_shape_product == output_shape_product)
            {
                AddInplace(op, 0, 0);
            }
        }

        else if (!OpInplacePass::shared_in_nodes(node))
        {
            if (auto op = std::dynamic_pointer_cast<ngraph::op::util::UnaryElementwiseArithmetic>(
                    node->get_op_ptr()))
            {
                AddInplace(op, 0, 0);
            }

            else if (auto op =
                         std::dynamic_pointer_cast<ngraph::op::util::BinaryElementwiseArithmetic>(
                             node->get_op_ptr()))
            {
                AddInplace(op, 0, 0);
            }

            else if (auto op = std::dynamic_pointer_cast<ngraph::op::Select>(node->get_op_ptr()))
            {
                AddInplace(op, 0, 1);
            }

            else if (node->get_op_type() == "AddN")
            {
                auto op = std::dynamic_pointer_cast<ngraph::op::GenericOp>(node->get_op_ptr());
                AddInplace(op, 0, 0);
            }

            else if (node->get_op_type() == "ApplyGradient")
            {
                auto op = std::dynamic_pointer_cast<ngraph::op::GenericOp>(node->get_op_ptr());
                AddInplace(op, 1, 0);
            }
        }
    }
    return true;
}

bool OpInplacePass::shared_in_nodes(std::shared_ptr<GNode>& node)
{
    for (auto& edge : node->get_in_edges())
    {
        auto in_node = edge->get_src();
        if (in_node->get_output_size() > 1)
        {
            return true;
        }
    }
    return false;
}
