// Microsoft (c) 2019, NNFusion Team

#include "op_inplace_pass.hpp"
#include "../gnode.hpp"
#include "../graph.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"

#include "nnfusion/core/ops/generic_op.hpp"

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

        if (node->get_op_type() == "Sum")
        {
            std::shared_ptr<ngraph::op::Sum> sum =
                std::dynamic_pointer_cast<ngraph::op::Sum>(node->get_op_ptr());

            ngraph::AxisSet reduce_axes = sum->get_reduction_axes();

            if (reduce_axes.empty())
            {
                auto op_annotations = sum->get_op_annotations();
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
                    sum->set_op_annotations(op_annotations);
                }
            }
        }

        if (node->get_op_type() == "Broadcast")
        {
            std::shared_ptr<ngraph::op::Broadcast> broadcast =
                std::static_pointer_cast<ngraph::op::Broadcast>(node->get_op_ptr());

            ngraph::AxisSet broadcast_axes = broadcast->get_broadcast_axes();

            if (broadcast_axes.empty())
            {
                auto op_annotations = broadcast->get_op_annotations();
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
                    broadcast->set_op_annotations(op_annotations);
                }
            }
        }

        if (node->get_op_type() == "Reduce")
        {
            std::shared_ptr<ngraph::op::Reduce> reduce =
                std::dynamic_pointer_cast<ngraph::op::Reduce>(node->get_op_ptr());

            ngraph::AxisSet reduce_axes = reduce->get_reduction_axes();

            if (reduce_axes.empty())
            {
                auto op_annotations = reduce->get_op_annotations();
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
                    reduce->set_op_annotations(op_annotations);
                }
            }
        }

        if (node->get_op_type() == "Max")
        {
            auto op = std::static_pointer_cast<ngraph::op::Max>(node->get_op_ptr());

            ngraph::AxisSet reduce_axes = op->get_reduction_axes();

            if (reduce_axes.empty())
            {
                auto op_annotations = op->get_op_annotations();
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
                    op->set_op_annotations(op_annotations);
                }
            }
        }

        if (node->get_op_type() == "Min")
        {
            auto op = std::static_pointer_cast<ngraph::op::Min>(node->get_op_ptr());

            ngraph::AxisSet reduce_axes = op->get_reduction_axes();

            if (reduce_axes.empty())
            {
                auto op_annotations = op->get_op_annotations();
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
                    op->set_op_annotations(op_annotations);
                }
            }
        }

        if (node->get_op_type() == "Abs")
        {
            auto op = std::static_pointer_cast<ngraph::op::Abs>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Acos")
        {
            auto op = std::static_pointer_cast<ngraph::op::Acos>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Asin")
        {
            auto op = std::static_pointer_cast<ngraph::op::Asin>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Atan")
        {
            auto op = std::static_pointer_cast<ngraph::op::Atan>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Ceiling")
        {
            auto op = std::static_pointer_cast<ngraph::op::Ceiling>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Cos")
        {
            auto op = std::static_pointer_cast<ngraph::op::Cos>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Cosh")
        {
            auto op = std::static_pointer_cast<ngraph::op::Cosh>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Exp")
        {
            auto op = std::static_pointer_cast<ngraph::op::Exp>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Floor")
        {
            auto op = std::static_pointer_cast<ngraph::op::Floor>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Log")
        {
            auto op = std::static_pointer_cast<ngraph::op::Log>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Sin")
        {
            auto op = std::static_pointer_cast<ngraph::op::Sin>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Sinh")
        {
            auto op = std::static_pointer_cast<ngraph::op::Sinh>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Sqrt")
        {
            auto op = std::static_pointer_cast<ngraph::op::Sqrt>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Tan")
        {
            auto op = std::static_pointer_cast<ngraph::op::Tan>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Tanh")
        {
            auto op = std::static_pointer_cast<ngraph::op::Tanh>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Power")
        {
            auto op = std::static_pointer_cast<ngraph::op::Power>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Subtract")
        {
            auto op = std::static_pointer_cast<ngraph::op::Subtract>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Divide")
        {
            auto op = std::static_pointer_cast<ngraph::op::Divide>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "DivNoNan")
        {
            auto op = std::static_pointer_cast<ngraph::op::DivNoNan>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Sign")
        {
            auto op = std::static_pointer_cast<ngraph::op::Sign>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Relu")
        {
            auto op = std::static_pointer_cast<ngraph::op::Relu>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Negative")
        {
            auto op = std::static_pointer_cast<ngraph::op::Negative>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Select")
        {
            auto op = std::static_pointer_cast<ngraph::op::Select>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
            if (op_annotations)
            {
                // pass-through
                op_annotations->add_in_place_oi_pair({0, 0, false});
            }
            else
            {
                op_annotations = std::make_shared<ngraph::op::util::OpAnnotations>();
                // pass-through
                op_annotations->add_in_place_oi_pair({0, 1, false});
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "ReluBackprop")
        {
            auto op = std::static_pointer_cast<ngraph::op::ReluBackprop>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Add")
        {
            auto op = std::static_pointer_cast<ngraph::op::Add>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "AddN")
        {
            auto op = std::static_pointer_cast<ngraph::op::GenericOp>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Multiply")
        {
            auto op = std::static_pointer_cast<ngraph::op::Multiply>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Minimum")
        {
            auto op = std::static_pointer_cast<ngraph::op::Minimum>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Maximum")
        {
            auto op = std::static_pointer_cast<ngraph::op::Maximum>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "Sigmoid")
        {
            auto op = std::static_pointer_cast<ngraph::op::Sigmoid>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }

        if (node->get_op_type() == "SigmoidBackprop")
        {
            auto op = std::static_pointer_cast<ngraph::op::SigmoidBackprop>(node->get_op_ptr());

            auto op_annotations = op->get_op_annotations();
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
                op->set_op_annotations(op_annotations);
            }
        }
    }
    return true;
}
