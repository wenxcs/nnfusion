// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(Pack).attr<int>("axis", 0).infershape([](ngraph::op::GenericOp& target_op) -> void {
    enforce(target_op.get_input_size() >= 1)
        << "Inputs of Pack operator should not be less than 1.";

    auto& input_shape_0 = target_op.get_input_shape(0);
    for (int i = 1; i < target_op.get_input_size(); i++)
    {
        auto& input_shape_n = target_op.get_input_shape(i);
        enforce(input_shape_0.size() == input_shape_n.size()) << "Shape dimension size not match.";
        for (int j = 0; j < input_shape_0.size(); j++)
        {
            enforce(input_shape_0[j] == input_shape_n[j]) << "Dimension " << j
                                                          << " in shapes must be equal.";
        }
    }

    int axis = target_op.localOpConfig.getRoot()["axis"];

    auto output_shape_0 = input_shape_0;
    output_shape_0.insert(output_shape_0.begin() + size_t(axis), target_op.get_input_size());

    target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
});