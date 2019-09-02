// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(Range).infershape([](ngraph::op::GenericOp& target_op) -> void {
    assert(target_op.get_input_size() == 3);
    auto ng_op_0 = target_op.get_argument(0); // start
    auto ng_op_1 = target_op.get_argument(1); // end
    auto ng_op_2 = target_op.get_argument(2); // step

    assert(ng_op_0->description() == "Constant");
    assert(ng_op_1->description() == "Constant");
    assert(ng_op_1->description() == "Constant");

    int start = std::dynamic_pointer_cast<ngraph::op::Constant>(ng_op_0)->get_vector<int>()[0];
    int end = std::dynamic_pointer_cast<ngraph::op::Constant>(ng_op_1)->get_vector<int>()[0];
    int step = std::dynamic_pointer_cast<ngraph::op::Constant>(ng_op_2)->get_vector<int>()[0];

    size_t len = (end - start) / step;
    LOG_INFO << "Range Infershape: " << start << ", " << end << ", " << step << ": " << len;
    ngraph::Shape output_shape_0 = {len};
    target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
});
