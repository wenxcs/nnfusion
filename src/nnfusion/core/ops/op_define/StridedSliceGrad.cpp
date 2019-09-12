// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/core/ops/generic_op.hpp"
// TODO: add StridedSliceGrad
// currently this is a hack impl for BERT_training
REGISTER_OP(StridedSliceGrad).infershape([](ngraph::op::GenericOp& target_op) -> void {
    ngraph::Shape output_shape_0 = {1, 256, 1024};
    target_op.set_output_type(0, element::f32, output_shape_0);
});
