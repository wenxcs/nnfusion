// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(Equal).infershape([](ngraph::op::GenericOp& target_op) -> void {
    const ngraph::Shape& input_shape_0 = target_op.get_input_shape(0);
    const ngraph::Shape& input_shape_1 = target_op.get_input_shape(1);

    ngraph::Shape output_shape_0;
    output_shape_0.push_back(input_shape_0.back());

    target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
});