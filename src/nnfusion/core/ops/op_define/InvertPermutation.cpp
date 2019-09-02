// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(InvertPermutation).infershape(ngraph::op::infershape::copy_shape_from_inputs);
