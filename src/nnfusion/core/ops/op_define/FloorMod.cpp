// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(FloorMod).infershape([](ngraph::op::GenericOp& target_op) -> void {
    target_op.set_output_type(0, target_op.get_input_element_type(0), target_op.get_input_shape(0));
});
