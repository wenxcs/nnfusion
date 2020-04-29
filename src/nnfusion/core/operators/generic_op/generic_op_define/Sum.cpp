// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Sum)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Sum>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();
        auto axes = _op->get_reduction_axes();

        auto input_shape = curr->get_input_shape(0);
        NNFUSION_CHECK(axes.size() >= 1);
        int min_axis = axes.size() + 1;
        for (auto& axis : axes)
            min_axis = min(min_axis, (int)axis);
        if (input_shape.size() - axes.size() == min_axis)
        {
            int batch = 1, sample = 1;
            for (int i = 0; i < min_axis; ++i)
                batch *= input_shape[i];
            for (int i = min_axis; i < input_shape.size(); ++i)
                sample *= input_shape[i];

            return op::create_code_from_template(
                "- input(\"input0\", [@batch@, @sample@]); output([@batch@], "
                "topi=topi.sum(args(\"input0\"), axis=1, keepdims=True));",
                {{"batch", batch}, {"sample", sample}});
        }
        else
        {
            // unimplemented case
            return "";
        }

    });