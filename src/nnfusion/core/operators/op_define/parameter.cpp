// Microsoft (c) 2019, NNFusion Team

#include "parameter.hpp"
using namespace std;
using namespace nnfusion::op;

Parameter::Parameter(const nnfusion::element::Type& element_type,
                     const nnfusion::Shape& shape,
                     const bool cacheable,
                     bool require_grad)
    : TensorOp("Parameter", element_type, shape)
    , m_cacheable(cacheable)
    , m_require_grad(require_grad)
{
}
