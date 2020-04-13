// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise cosine operation.
        class Erf : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a error function operation.
            Erf();
        };
    }
}