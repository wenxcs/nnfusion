// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/core/ops/generic_op.hpp"

namespace ngraph
{
    namespace op
    {
        std::unordered_map<std::string, ngraph::op::OpConfig>& get_op_configs()
        {
            static std::unordered_map<std::string, ngraph::op::OpConfig> __op_configs;
            return __op_configs;
        }
    } // namespace op
} // namespace ngraph
