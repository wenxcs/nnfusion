// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Broadcast)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = static_pointer_cast<nnfusion::op::Broadcast>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        // Input shape should have the same rank with output shape.
        // For each dimension,
        //   1. input shape has the same value with output shape, or
        //   2. input shape has the value "1" at the broadcast axes.
        nnfusion::Shape input_shape = gnode->get_output_shape(0);
        nnfusion::Shape output_shape = gnode->get_output_shape(0);
        const auto& axes = op->get_broadcast_axes();
        for (auto axis : axes)
        {
            input_shape[axis] = 1;
        }

        bool memcpy_annotation = (shape_size(input_shape) == shape_size(output_shape));
        if (output_shape.size() == 0)
        {
            output_shape.push_back(1);
        }

        auto expression = op::create_code_from_template(
            R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.broadcast_to(args("input0"), @output_shape@)); )",
            {{"input_shape", vector_to_string(input_shape)},
             {"output_shape", vector_to_string(output_shape)}});

        return expression + (memcpy_annotation ? " ## @annotation: memcpy" : "");
    });
