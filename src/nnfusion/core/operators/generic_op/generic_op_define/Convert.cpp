// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/common/type/element_type.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Convert)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = static_pointer_cast<nnfusion::op::Convert>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        std::string dtype;
        bool ret = element::Type::nnfusion_element_type_to_dtype_string(
            op->get_convert_element_type(), dtype);
        NNFUSION_CHECK(ret == true) << "cast type is not supported: "
                                    << op->get_convert_element_type().c_type_string();

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@); output(@input_shape@, topi=topi.cast(args("input0"), dtype="@dtype@")); )",
            {{"input_shape", vector_to_string(gnode->get_input_shape(0))}, {"dtype", dtype}});
    });