// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(Range).attr<int>("start").attr<int>("limit").attr<int>("delta").infershape(
    [](ngraph::op::GenericOp& target_op) -> void {

        float start = target_op.localOpConfig.getRoot()["start"];
        float limit = target_op.localOpConfig.getRoot()["limit"];
        float delta = target_op.localOpConfig.getRoot()["delta"];
        int num = (int)((limit - start + delta - 1) / delta);

        ngraph::Shape output_shape_0;
        output_shape_0.push_back(num);

        target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
    });
