// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"

REGISTER_OP(Broadcast)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Broadcast>(curr->get_op_ptr());
        CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        // auto& axes = _op->get_broadcast_axes();

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.broadcast_to(args("input0"), @output_shape@)); )",
            {{"input_shape", vector_to_string(curr->get_input_shape(0))},
             {"output_shape", vector_to_string(curr->get_output_shape(0))}});
    });
