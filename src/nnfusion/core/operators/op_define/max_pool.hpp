// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Batched max pooling operation, with optional padding and window stride.
        class MaxPool : public Op
        {
        public:
            /// \brief Constructs a batched max pooling operation.
            ///
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            /// \param padding_below The below-padding shape.
            /// \param padding_above The above-padding shape.
            MaxPool(const ngraph::Shape& window_shape,
                    const ngraph::Strides& window_movement_strides,
                    const ngraph::Shape& padding_below,
                    const ngraph::Shape& padding_above);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            /// \brief Constructs a batched, unpadded max pooling operation (i.e., all padding shapes are set to 0).
            ///
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            MaxPool(const ngraph::Shape& window_shape,
                    const ngraph::Strides& window_movement_strides);

            /// \brief Constructs an unstrided batched max pooling operation (i.e., all window movement strides are 1 and all padding shapes are set to 0).
            ///
            /// \param window_shape The window shape.
            MaxPool(const ngraph::Shape& window_shape);

            /// \return The window shape.
            const ngraph::Shape& get_window_shape() const { return m_window_shape; }
            /// \return The window movement strides.
            const ngraph::Strides& get_window_movement_strides() const
            {
                return m_window_movement_strides;
            }
            /// \return The below-padding shape.
            const ngraph::Shape& get_padding_below() const { return m_padding_below; }
            /// \return The above-padding shape.
            const ngraph::Shape& get_padding_above() const { return m_padding_above; }
        protected:
            ngraph::Shape m_window_shape;
            ngraph::Strides m_window_movement_strides;
            ngraph::Shape m_padding_below;
            ngraph::Shape m_padding_above;
        };

        class MaxPoolBackprop : public Op
        {
        public:
            MaxPoolBackprop(const ngraph::Shape& window_shape,
                            const ngraph::Strides& window_movement_strides,
                            const ngraph::Shape& padding_below,
                            const ngraph::Shape& padding_above,
                            const std::shared_ptr<MaxPool>& forward_op = nullptr);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            const ngraph::Shape& get_window_shape() const { return m_window_shape; }
            const ngraph::Strides& get_window_movement_strides() const
            {
                return m_window_movement_strides;
            }
            const ngraph::Shape& get_padding_below() const { return m_padding_below; }
            const ngraph::Shape& get_padding_above() const { return m_padding_above; }
            /// \return A pointer to the corresponding `MaxPool` forward prop op. This may be
            ///         `nullptr` if no such pointer was provided at construction time, or if the
            ///         forward op has been freed due to graph rewriting.
            std::shared_ptr<MaxPool> get_forward_op() const;

        protected:
            ngraph::Shape m_window_shape;
            ngraph::Strides m_window_movement_strides;
            ngraph::Shape m_padding_below;
            ngraph::Shape m_padding_above;
            std::weak_ptr<MaxPool> m_forward_op;
        };
    }
}
