// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise square root operation.
        class Rsqrt : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a square operation.
            Rsqrt();
        };
    }
}