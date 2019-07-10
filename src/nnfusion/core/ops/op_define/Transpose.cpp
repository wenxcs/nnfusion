// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(Transpose)
    .attr<std::vector<int>>("axes_order")
    .infershape([](ngraph::op::GenericOp& target_op) -> void {
        auto& shape_0 = target_op.get_input_shape(0);
        auto& axes_order = target_op.localOpConfig.getRoot()["axes_order"];
        assert(axes_order.size() == shape_0.size());
        ngraph::Shape output_shape_0;
        for (int i = 0; i < axes_order.size(); ++i)
            output_shape_0.push_back(shape_0[axes_order[i]]);
        target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
    });
