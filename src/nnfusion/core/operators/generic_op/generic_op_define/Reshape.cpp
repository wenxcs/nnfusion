// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Reshape)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = static_pointer_cast<nnfusion::op::Reshape>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        std::string expression;
        bool memcpy_annotation = false;
        if (op->get_is_transpose())
        {
            const auto& input_order = op->get_input_order();
            expression = op::create_code_from_template(
                R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.transpose(args("input0"), axes=@input_order@)); )",
                {{"input_shape", vector_to_string(gnode->get_input_shape(0))},
                 {"output_shape", vector_to_string(gnode->get_output_shape(0))},
                 {"input_order", vector_to_string(input_order)}});
            for (int i = 0; i < input_order.size(); ++i)
                if (input_order[i] != i)
                    break;
                else if (i + 1 == input_order.size())
                    memcpy_annotation = true;
        }
        else
        {
            // For cases with "is_transpose==false", the ReshapeMemcpy kernel will be selected.
            size_t total_size = 1L;
            for (auto it : gnode->get_input_shape(0))
                total_size *= it;
            expression = op::create_code_from_template(
                R"( - input("input0", @input_shape@); output(@output_shape@, lambda i: args("input0")[i]); )",
                {{"input_shape", "[" + std::to_string(total_size) + "]"},
                 {"output_shape", "[" + std::to_string(total_size) + "]"}});
            memcpy_annotation = true;
        }
        return expression + (memcpy_annotation ? " ## @annotation: memcpy" : "");
    });
