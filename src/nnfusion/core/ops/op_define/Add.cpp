// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(Add).infershape([](ngraph::op::GenericOp& target_op) -> void {
    assert(2 == target_op.get_input_size());
    auto& shape_0 = target_op.get_input_shape(0);
    auto& shape_1 = target_op.get_input_shape(0);
    assert(shape_0.size() == shape_1.size());
    ngraph::Shape output_shape_0;
    for (int i = 0; i < shape_0.size(); ++i)
    {
        if (shape_0[i] != shape_1[i])
            assert(shape_0[i] == 1 || shape_1[i] == 1);
        output_shape_0.push_back(std::max(shape_0[i], shape_1[i]));
    }
    target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
});
