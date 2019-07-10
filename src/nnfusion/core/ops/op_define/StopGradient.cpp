// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(StopGradient).infershape(ngraph::op::infershape::copy_shape_from_inputs);
