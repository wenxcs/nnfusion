// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Convolution)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Convolution>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        const auto& dilation = _op->get_window_dilation_strides();
        const auto& stride = _op->get_window_movement_strides();
        const auto& padding_below = _op->get_padding_below();
        const auto& padding_above = _op->get_padding_above();
        int64_t padding[] = {
            padding_below[1], padding_below[0], padding_above[1], padding_above[0]};

        return op::create_code_from_template(
            R"( - input("input0", @input_shape_0@); input("input1", @input_shape_1@); output(@output_shape@, topi=topi.nn.conv2d_nchw(args("input0"), args("input1"), stride=@stride@, padding=@padding@, dilation=@dilation@)); )",
            {{"input_shape_0", vector_to_string(curr->get_input_shape(0))},
             {"input_shape_1", vector_to_string(curr->get_input_shape(1))},
             {"output_shape", vector_to_string(curr->get_output_shape(0))},
             {"stride", vector_to_string(stride)},
             {"padding", vector_to_string(padding)},
             {"dilation", vector_to_string(dilation)}});
    });