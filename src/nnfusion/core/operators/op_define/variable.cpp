// Microsoft (c) 2019, NNFusion Team

#include "variable.hpp"
using namespace nnfusion::op;

Variable::Variable(const nnfusion::element::Type& element_type, const nnfusion::Shape& shape)
    : TensorOp("Variable", element_type, shape)
{
}
