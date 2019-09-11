// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

// TODO: Need to be more specific

REGISTER_OP(ApplyGradient)
    .attr<float>("learning_rate", 0.001)
    .infershape([](ngraph::op::GenericOp& target_op) -> void {

        enforce(target_op.get_input_size() == 2) << "Inputs of ApplyGradient operator should be 2.";

        auto& weight_tensor = target_op.get_input_shape(0);
        auto& gradient_tensor = target_op.get_input_shape(1);

        enforce(weight_tensor.size() == gradient_tensor.size())
            << "The two inputs should have the same dimentions.";
        for (int j = 0; j < weight_tensor.size(); j++)
        {
            enforce(weight_tensor[j] == gradient_tensor[j]) << "Dimension " << j
                                                            << " in shapes must be equal.";
        }

        ngraph::Shape output_shape_0(weight_tensor);
        target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
    });
