// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Concat)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = static_pointer_cast<nnfusion::op::Concat>(gnode->get_op_ptr());
        CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        size_t axis = op->get_concatenation_axis();

        std::stringstream expression;
        expression << "- ";
        size_t num_inputs = gnode->get_input_size();
        for (size_t i = 0; i < num_inputs; ++i)
        {
            expression << "input(\"input" << i << "\", "
                       << vector_to_string(gnode->get_input_shape(i)) << "); ";
        }
        expression << "output(" << vector_to_string(gnode->get_output_shape(0))
                   << ", topi=topi.concatenate([";
        for (size_t i = 0; i < num_inputs - 1; ++i)
        {
            expression << "args(\"input" << i << "\"), ";
        }
        expression << "args(\"input" << (num_inputs - 1) << "\")], ";
        expression << "axis=" << axis << "));";

        return expression.str();

    });
