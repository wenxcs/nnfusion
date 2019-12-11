// Microsoft (c) 2019, NNFusion Team

#include "not.hpp"

using namespace nnfusion::op;
using namespace std;

Not::Not()
    : Op("Not")
{
}

// TODO(amprocte): Update this to allow only boolean, for consistency with logical binops.
void Not::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto args_et_pshape = validate_and_infer_elementwise_args(gnode);
    ngraph::element::Type& args_et = std::get<0>(args_et_pshape);
    ngraph::PartialShape& args_pshape = std::get<1>(args_et_pshape);

    set_output_type_and_shape(gnode, 0, args_et, args_pshape);
}
