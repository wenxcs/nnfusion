// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/arithmetic_reduction.hpp"
#include "nnfusion/core/operators/util/binary_elementwise_logical.hpp"

namespace nnfusion
{
    namespace op
    {
        class ReduceAny : public ArithmeticReduction
        {
        public:
            /// \brief Constructs a logical-or operation.
            ///
            ReduceAny(const nnfusion::AxisSet& reduction_axes);
        };

        /// \brief Elementwise logical-or operation.
        ///
        class Or : public BinaryElementwiseLogical
        {
        public:
            /// \brief Constructs a logical-or operation.
            Or();

        protected:
            virtual bool is_commutative() override { return true; }
        };
    }
}
