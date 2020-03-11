// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Slice)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = static_pointer_cast<nnfusion::op::Slice>(gnode->get_op_ptr());
        CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.strided_slice(args("input0"), begin=@begin@, end=@end@, strides=@strides@)); )",
            {{"input_shape", vector_to_string(gnode->get_input_shape(0))},
             {"output_shape", vector_to_string(gnode->get_output_shape(0))},
             {"begin", vector_to_string(op->get_lower_bounds())},
             {"end", vector_to_string(op->get_upper_bounds())},
             {"strides", vector_to_string(op->get_strides())}});

    });
