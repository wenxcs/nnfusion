// Microsoft (c) 2019, NNFusion Team

#include "binary_elementwise_logical.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

BinaryElementwiseLogical::BinaryElementwiseLogical(const string& node_type)
    : Op(node_type)
{
}

void BinaryElementwiseLogical::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto args_et_pshape = validate_and_infer_elementwise_args(gnode);
    ngraph::element::Type& args_et = std::get<0>(args_et_pshape);
    ngraph::PartialShape& args_pshape = std::get<1>(args_et_pshape);

    OP_VALIDATION(this, args_et.is_dynamic() || args_et == ngraph::element::boolean)
        << "Operands for logical operators must have boolean element type but have element type "
        << args_et << ".";

    gnode->set_output_type_and_shape(0, ngraph::element::boolean, args_pshape);
}
