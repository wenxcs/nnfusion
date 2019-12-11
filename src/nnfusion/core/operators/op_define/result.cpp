// Microsoft (c) 2019, NNFusion Team

#include "result.hpp"
#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

Result::Result()
    : Op("Result")
{
}

void Result::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    OP_VALIDATION(this, gnode->get_input_size() == 1) << "Argument has " << gnode->get_input_size()
                                                      << " outputs (1 expected).";

    // always borrow the placement conf even the default one
    set_placement(gnode->get_in_edge(0)->get_src()->get_op_ptr()->get_placement());
    set_output_type_and_shape(
        gnode, 0, gnode->get_input_element_type(0), gnode->get_input_partial_shape(0));
}
