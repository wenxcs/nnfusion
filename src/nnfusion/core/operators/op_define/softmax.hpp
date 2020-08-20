// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Softmax operation.
        ///
        class Softmax : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a softmax operation.
            ///
            /// \param axes The axis positions (0-based) on which to calculate the softmax.
            Softmax(const nnfusion::AxisSet& axes,
                    bool in_log_space =
                        false); // Current kernel doesn't follow the axes, but the last dim

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            const nnfusion::AxisSet& get_axes() const { return m_axes; }
            const bool is_in_log_space() const { return m_in_log_space; }
        private:
            nnfusion::AxisSet m_axes;
            bool m_in_log_space;
        };

        /// \brief Softmax operation.
        ///
        class SoftmaxGrad : public Op
        {
        public:
            /// \brief Constructs a softmax grad operation.
            ///
            /// \param axes The axis positions (0-based) on which to calculate the softmax.
            SoftmaxGrad(const nnfusion::AxisSet& axes, bool in_log_space = false);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            const nnfusion::AxisSet& get_axes() const { return m_axes; }
            const bool is_in_log_space() const { return m_in_log_space; }
        private:
            nnfusion::AxisSet m_axes;
            bool m_in_log_space;
        };
    }
}
