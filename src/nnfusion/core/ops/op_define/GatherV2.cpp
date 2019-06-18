// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(GatherV2).attr<int>("axis", 0).infershape([](ngraph::op::GenericOp& target_op) -> void {
    assert(target_op.get_input_size() == 2);
    const ngraph::Shape& input_shape_0 = target_op.get_input_shape(0);
    const ngraph::Shape& input_shape_1 = target_op.get_input_shape(1);

    int axis = target_op.localOpConfig.getRoot()["axis"];

    ngraph::Shape output_shape_0;
    for (int i = 0; i < axis; ++i)
        output_shape_0.push_back(input_shape_0[i]);
    for (int i = 0; i < input_shape_1.size(); ++i)
        output_shape_0.push_back(input_shape_1[i]);
    for (int i = axis + 1; i < input_shape_0.size(); ++i)
        output_shape_0.push_back(input_shape_0[i]);

    target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
});
