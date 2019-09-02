// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(Tile).infershape([](ngraph::op::GenericOp& target_op) -> void {
    assert(target_op.get_input_size() == 2);
    auto ng_op = target_op.get_argument(1);
    assert(ng_op->description() == "Constant");
    auto out_shape = std::dynamic_pointer_cast<ngraph::op::Constant>(ng_op)->get_vector<int>();
    ngraph::Shape output_shape_0;
    for (auto& it : out_shape)
        output_shape_0.push_back(it);
    target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
});
