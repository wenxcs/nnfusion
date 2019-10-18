// Microsoft (c) 2019, NNFusion Team

#include "ngraph/serializer.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(CrossEntropyAvgLossWithLabels).infershape([](ngraph::op::GenericOp& target_op) -> void {
    CHECK(2 == target_op.get_input_size());
    auto& shape_0 = target_op.get_input_shape(0);
    auto& shape_1 = target_op.get_input_shape(1);

    CHECK(shape_0.size() == 2 && shape_1.size() == 1 && shape_0[0] == shape_1[0]);

    target_op.set_output_type(0, target_op.get_input_element_type(0), shape_1);
});
