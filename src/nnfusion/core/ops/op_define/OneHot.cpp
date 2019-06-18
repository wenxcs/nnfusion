// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(OneHot)
    .attr<int>("axis", -1)
    .attr<int>("depth")
    .attr<ngraph::op::OpConfig::any>("off_value")
    .attr<ngraph::op::OpConfig::any>("on_value")
    .infershape([](ngraph::op::GenericOp& target_op) -> void {
        assert(1 == target_op.get_input_size());
        auto& shape_0 = target_op.get_input_shape(0);
        int depth = target_op.localOpConfig.getRoot()["depth"];
        int axis = target_op.localOpConfig.getRoot()["axis"];
        if (axis == -1)
            axis = shape_0.size() - 1;
        ngraph::Shape output_shape_0;
        for (int i = 0; i < axis; ++i)
            output_shape_0.push_back(shape_0[i]);
        output_shape_0.push_back(depth);
        for (int i = axis + 1; i < shape_0.size(); ++i)
            output_shape_0.push_back(shape_0[i]);
        target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
    });
