// Microsoft (c) 2019, NNFusion Team

#include "op_inplace_pass.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/reduce.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/op_define/result.hpp"
#include "nnfusion/core/operators/op_define/select.hpp"
#include "nnfusion/core/operators/util/arithmetic_reduction.hpp"
#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::op;
using namespace nnfusion::pass::graph;

bool OpInplacePass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    for (auto node : graph->get_nodes())
    {
        if (auto op = std::dynamic_pointer_cast<ArithmeticReduction>(node->get_op_ptr()))
        {
            ngraph::AxisSet reduce_axes = op->get_reduction_axes();
            auto input_shape_product = shape_size(node->get_input_shape(0));
            auto output_shape_product = shape_size(node->get_output_shape(0));
            if (reduce_axes.empty() || input_shape_product == output_shape_product)
            {
                AddInplace(op, 0, 0, true);
            }
        }

        // add inplace tag for reshape op if !op->get_is_transpose() || op element num < 2
        else if (auto op = std::dynamic_pointer_cast<Reshape>(node->get_op_ptr()))
        {
            ngraph::Shape result_shape = node->get_output_shape(0);
            size_t result_shape_product = ngraph::shape_size(result_shape);

            if (!op->get_is_transpose() || result_shape_product < 2)
            {
                AddInplace(op, 0, 0, true);
            }
        }

        else if (auto op = std::dynamic_pointer_cast<Result>(node->get_op_ptr()))
        {
            AddInplace(op, 0, 0, true);
        }

        else if (auto op = std::dynamic_pointer_cast<Broadcast>(node->get_op_ptr()))
        {
            ngraph::AxisSet broadcast_axes = op->get_broadcast_axes();
            auto input_shape_product = shape_size(node->get_input_shape(0));
            auto output_shape_product = shape_size(node->get_output_shape(0));

            if (broadcast_axes.empty() || input_shape_product == output_shape_product)
            {
                AddInplace(op, 0, 0, true);
            }
        }

        else if (auto op = std::dynamic_pointer_cast<Reduce>(node->get_op_ptr()))
        {
            ngraph::AxisSet reduce_axes = op->get_reduction_axes();
            auto input_shape_product = shape_size(node->get_input_shape(0));
            auto output_shape_product = shape_size(node->get_output_shape(0));
            if (reduce_axes.empty() || input_shape_product == output_shape_product)
            {
                AddInplace(op, 0, 0, true);
            }
        }

        if (auto op = std::dynamic_pointer_cast<ElementwiseArithmetic>(node->get_op_ptr()))
        {
            AddInplace(op, 0, 0, true);
        }

        else if (auto op = std::dynamic_pointer_cast<Select>(node->get_op_ptr()))
        {
            AddInplace(op, 0, 1, true);
        }

        else if (node->get_op_type() == "AddN")
        {
            auto op = std::dynamic_pointer_cast<GenericOp>(node->get_op_ptr());
            AddInplace(op, 0, 0, true);
        }

        else if (node->get_op_type() == "ApplyGradient")
        {
            auto op = std::dynamic_pointer_cast<GenericOp>(node->get_op_ptr());
            AddInplace(op, 0, 0, true);
        }
    }
    return true;
}
