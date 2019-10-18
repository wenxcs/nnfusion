// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(Tile).infershape([](ngraph::op::GenericOp& target_op) -> void {
    CHECK(target_op.get_input_size() == 2);
    auto& input_shape_0 = target_op.get_input_shape(0);
    auto ng_op = target_op.get_argument(1);
    CHECK(ng_op->description() == "Constant")
        << "We only accept the Tile input \"multiples\" as Constant.";
    ///\todo multiples must be int32 or int64, we use int32 in this case, currently we ignore int64
    auto multiples = std::dynamic_pointer_cast<ngraph::op::Constant>(ng_op)->get_vector<int64_t>();
    CHECK(input_shape_0.size() == multiples.size());
    ngraph::Shape output_shape_0(multiples.size());
    for (int i = 0; i < multiples.size(); i++)
        output_shape_0[i] = multiples[i] * input_shape_0[i];
    target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
});