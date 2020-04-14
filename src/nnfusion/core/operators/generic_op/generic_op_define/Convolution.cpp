// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Convolution)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Convolution>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        // Not implemented
        return "";
    });
