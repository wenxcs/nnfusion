// Microsoft (c) 2019, NNFusion Team

#include "tensor_op.hpp"
#include "nnfusion/core/graph/gnode.hpp"
using namespace std;
using namespace nnfusion::op;

TensorOp::TensorOp(const std::string& node_type,
                   const nnfusion::element::Type& element_type,
                   const nnfusion::Shape& shape)
    : Op(node_type)
    , m_shape(shape)
    , m_element_type(element_type)
{
}

void TensorOp::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    Op::validate_and_infer_types(gnode);

    gnode->set_output_type_and_shape(0, m_element_type, m_shape);
}
