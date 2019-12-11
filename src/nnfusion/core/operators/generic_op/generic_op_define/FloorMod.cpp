// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(FloorMod).infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
    gnode->get_op_ptr()->set_output_type_and_shape(
        gnode, 0, gnode->get_input_element_type(0), gnode->get_input_shape(0));
});
