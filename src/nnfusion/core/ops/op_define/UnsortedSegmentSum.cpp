// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

/*
Computes a tensor such that \(output[i] = {j...} data[j...]\) where 
the sum is over tuples j... such that segment_ids[j...] == i. 
Unlike SegmentSum, segment_ids need not be sorted and need not cover 
all values in the full range of valid values.
If the sum is empty for a given segment ID i, output[i] = 0. 
If the given segment ID i is negative, the value is dropped and 
will not be added to the sum of the segment.
num_segments should equal the number of distinct segment IDs.
*/

REGISTER_OP(UnsortedSegmentSum).infershape([](ngraph::op::GenericOp& target_op) -> void {
    GENERIC_OP_LOGGING();
    enforce(target_op.get_input_size() == 3) << "Inputs of UnsortedSegmentSum should be 3.";
    // Outshape is as same as input data, (except the first one);
    auto ng_group = target_op.get_argument(1);
    auto ng_seg = target_op.get_argument(2);
    enforce(ng_seg->description() == "Constant")
        << "We only accept the sgements number as Constant.";
    auto& shape_0 = target_op.get_input_shape(0);
    auto& shape_1 = target_op.get_input_shape(1);
    auto& shape_2 = target_op.get_input_shape(2);
    auto constop = std::dynamic_pointer_cast<ngraph::op::Constant>(ng_seg);
    auto seg_num = constop->get_vector<int>();
    enforce(shape_0.size() > 0 && shape_1.size() == 1 && seg_num.size() == 1)
        << "Only support 1-D sgments." << shape_0 << shape_1 << shape_2;
    ngraph::Shape output_shape(shape_0);
    // Output: Has same shape as data,
    // except for the first segment_ids.rank dimensions,
    // which are replaced with a single dimension which has size num_segments.
    output_shape[0] = seg_num[0];
    target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape);
});