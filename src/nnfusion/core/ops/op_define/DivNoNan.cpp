// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

// TODO: Need to be more specific

REGISTER_OP(DivNoNan).infershape([](ngraph::op::GenericOp& target_op) -> void {
    assert(target_op.get_input_shape(0) == target_op.get_input_shape(1));
    target_op.set_output_type(0, target_op.get_input_element_type(0), target_op.get_input_shape(0));
});
