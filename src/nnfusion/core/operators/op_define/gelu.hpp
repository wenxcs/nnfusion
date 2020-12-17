// Microsoft (c) 2020, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise cosine operation.
        class Gelu : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a gelu operation.
            Gelu();
        };
    }
}
