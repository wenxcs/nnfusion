// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(AddN).attr<ngraph::op::OpConfig::any>("T").infershape(
    [](ngraph::op::GenericOp& target_op) -> void {
        // enforce is like assert, but when thing goes wrong, it will print error message.
        enforce(target_op.get_input_size() >= 2)
            << "Inputs of AddN operator should not be less than 2.";

        auto& shape_0 = target_op.get_input_shape(0);
        for (int i = 1; i < target_op.get_input_size(); i++)
        {
            auto& shape_n = target_op.get_input_shape(i);
            enforce(shape_0.size() == shape_n.size()) << "Shape dimension size not match.";
            for (int j = 0; j < shape_0.size(); j++)
            {
                enforce(shape_0[j] == shape_n[j]) << "Dimension " << j
                                                  << " in shapes must be equal.";
            }
        }

        ngraph::Shape output_shape_0(shape_0);
        target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
    });
