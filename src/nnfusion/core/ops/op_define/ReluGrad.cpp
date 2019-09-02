// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(ReluGrad).infershape([](ngraph::op::GenericOp& target_op) -> void {
    assert(target_op.get_input_size() == 2);
    auto shape_0 = target_op.get_input_shape(0);
    auto shape_1 = target_op.get_input_shape(1);
    assert(shape_0 == shape_1);
    target_op.set_output_type(0, target_op.get_input_element_type(0), shape_0);
});
