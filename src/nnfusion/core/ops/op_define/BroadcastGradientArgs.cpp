// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

// TODO: Need to be more specific

REGISTER_OP(BroadcastGradientArgs).infershape([](ngraph::op::GenericOp& target_op) -> void {
    ngraph::Shape output_shape = {};
    target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape);
    target_op.set_output_type(1, target_op.get_input_element_type(0), output_shape);
});
