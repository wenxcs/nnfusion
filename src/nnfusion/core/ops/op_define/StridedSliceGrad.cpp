// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/core/ops/generic_op.hpp"
// TODO: add StridedSliceGrad
// currently this is a hack impl for BERT_training
REGISTER_OP(StridedSliceGrad)
    .attr<int>("begin_mask", 0)
    .attr<int>("end_mask", 0)
    .attr<int>("ellipsis_mask", 0)
    .attr<int>("new_axis_mask", 0)
    .attr<int>("shrink_axis_mask", 0)
    .infershape([](ngraph::op::GenericOp& target_op) -> void {
        assert(target_op.get_input_size() == 5);

        int begin_mask = target_op.localOpConfig.getRoot()["begin_mask"];
        int end_mask = target_op.localOpConfig.getRoot()["end_mask"];
        int ellipsis_mask = target_op.localOpConfig.getRoot()["ellipsis_mask"];
        int new_axis_mask = target_op.localOpConfig.getRoot()["new_axis_mask"];
        int shrink_axis_mask = target_op.localOpConfig.getRoot()["shrink_axis_mask"];
        // TODO: handle the cases that these attrs are not zeros
        enforce(begin_mask == 0 && end_mask == 0 && ellipsis_mask == 0 && new_axis_mask == 0 &&
                shrink_axis_mask == 0)
            << "do not support mast attributes yet!";

        // Set output size
        auto x = target_op.get_argument(0);
        auto x_value = std::dynamic_pointer_cast<ngraph::op::Constant>(x)->get_vector<int32_t>();
        const ngraph::Shape& input_shape_0 = target_op.get_input_shape(0);
        int x_size = input_shape_0[0];
        enforce(x_size == x_value.size());

        //Bert Defaut: ngraph::Shape output_shape_0 = {1, 256, 1024};
        ngraph::Shape output_shape_0;
        for (int i = 0; i < x_size; ++i)
            output_shape_0.push_back(x_value[i]);

        target_op.set_output_type(0, element::f32, output_shape_0);
    });
