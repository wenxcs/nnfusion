// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../op.hpp"
#include "ngraph/coordinate_diff.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Batched convolution operation, with optional window dilation and stride.
        ///
        class Convolution : public Op
        {
        public:
            /// \brief Constructs a batched convolution operation.
            ///
            /// \param window_movement_strides The window movement strides.<br>
            /// `[f]`
            /// \param window_dilation_strides The window dilation strides.<br>
            /// `[f]`
            /// \param padding_below The padding-below sizes.<br>
            /// `[f]`
            /// \param padding_above The padding-above sizes.<br>
            /// `[f]`
            /// \param data_dilation_strides The data dilation strides.<br>
            /// `[f]`
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution(const ngraph::Strides& window_movement_strides,
                        const ngraph::Strides& window_dilation_strides,
                        const ngraph::CoordinateDiff& padding_below,
                        const ngraph::CoordinateDiff& padding_above,
                        const ngraph::Strides& data_dilation_strides);

            /// \brief Constructs a batched convolution operation with no data dilation (i.e., all data dilation strides are 1).
            ///
            /// \param window_movement_strides The window movement strides.<br>
            /// `[f]`
            /// \param window_dilation_strides The window dilation strides.<br>
            /// `[f]`
            /// \param padding_below The padding-below sizes.<br>
            /// `[f]`
            /// \param padding_above The padding-above sizes.<br>
            /// `[f]`
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution(const ngraph::Strides& window_movement_strides,
                        const ngraph::Strides& window_dilation_strides,
                        const ngraph::CoordinateDiff& padding_below,
                        const ngraph::CoordinateDiff& padding_above);

            /// \brief Constructs a batched convolution operation with no padding or data dilation (i.e., padding above and below are 0 everywhere, and all data dilation strides are 1).
            ///
            /// \param window_movement_strides The window movement strides.<br>
            /// `[f]`
            /// \param window_dilation_strides The window dilation strides.<br>
            /// `[f]`
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution(const ngraph::Strides& window_movement_strides,
                        const ngraph::Strides& window_dilation_strides);

            /// \brief Constructs a batched convolution operation with no window dilation, padding, or data dilation (i.e., padding above and below are 0 everywhere, and all window/data dilation strides are 1).
            ///
            /// \param window_movement_strides The window movement strides.<br>
            /// `[f]`
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution(const ngraph::Strides& window_movement_strides);

            /// \brief Constructs a batched convolution operation with no window dilation or movement stride (i.e., padding above and below are 0 everywhere, and all window/data dilation strides and window movement strides are 1).
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution();

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            /// \return The window movement strides.
            const ngraph::Strides& get_window_movement_strides() const
            {
                return m_window_movement_strides;
            }
            /// \return The window dilation strides.
            const ngraph::Strides& get_window_dilation_strides() const
            {
                return m_window_dilation_strides;
            }
            /// \return The padding-below sizes (possibly negative).
            const ngraph::CoordinateDiff& get_padding_below() const { return m_padding_below; }
            /// \return The padding-above sizes (possibly negative).
            const ngraph::CoordinateDiff& get_padding_above() const { return m_padding_above; }
            /// \return The input data dilation strides.
            const ngraph::Strides& get_data_dilation_strides() const
            {
                return m_data_dilation_strides;
            }
            /// \return The default value for Convolution.

        protected:
            ngraph::Strides m_window_movement_strides;
            ngraph::Strides m_window_dilation_strides;
            ngraph::CoordinateDiff m_padding_below;
            ngraph::CoordinateDiff m_padding_above;
            ngraph::Strides m_data_dilation_strides;

        private:
            static ngraph::Strides default_strides(const Op* op,
                                                   const ngraph::PartialShape& data_batch_shape,
                                                   const ngraph::PartialShape& filters_shape);
            static ngraph::CoordinateDiff
                default_padding(const Op* op,
                                const ngraph::PartialShape& data_batch_shape,
                                const ngraph::PartialShape& filters_shape);
        };

        /// \brief Data batch backprop for batched convolution operation.
        class ConvolutionBackpropData : public Op
        {
        public:
            /// \brief Constructs a batched-convolution data batch-backprop operation.
            ///
            /// \param data_batch_shape The shape of the data batch from forward-prop.
            /// \param window_movement_strides_forward The window movement strides from forward-prop.
            /// \param window_dilation_strides_forward The window dilation strides from forward-prop.
            /// \param padding_below_forward The padding-below sizes from forward-prop.
            /// \param padding_above_forward The padding-above sizes from forward-prop.
            /// \param data_dilation_strides_forward The data dilation strides from forward-prop.
            ConvolutionBackpropData(const ngraph::Shape& data_batch_shape,
                                    const ngraph::Strides& window_movement_strides_forward,
                                    const ngraph::Strides& window_dilation_strides_forward,
                                    const ngraph::CoordinateDiff& padding_below_forward,
                                    const ngraph::CoordinateDiff& padding_above_forward,
                                    const ngraph::Strides& data_dilation_strides_forward);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            /// \return The data batch shape.
            const ngraph::Shape& get_data_batch_shape() const { return m_data_batch_shape; }
            /// \return The window movement strides from the forward prop.
            const ngraph::Strides& get_window_movement_strides_forward() const
            {
                return m_window_movement_strides_forward;
            }
            /// \return The window dilation strides from the forward prop.
            const ngraph::Strides& get_window_dilation_strides_forward() const
            {
                return m_window_dilation_strides_forward;
            }
            /// \return The padding-below sizes (possibly negative) from the forward prop.
            const ngraph::CoordinateDiff& get_padding_below_forward() const
            {
                return m_padding_below_forward;
            }
            /// \return The padding-above sizes (possibly negative) from the forward prop.
            const ngraph::CoordinateDiff& get_padding_above_forward() const
            {
                return m_padding_above_forward;
            }
            /// \return The input data dilation strides from the forward prop.
            const ngraph::Strides& get_data_dilation_strides_forward() const
            {
                return m_data_dilation_strides_forward;
            }

            /// \return The window movement strides for the backward prop.
            const ngraph::Strides& get_window_movement_strides_backward() const
            {
                return m_window_movement_strides_backward;
            }
            /// \return The window dilation strides for the backward prop.
            const ngraph::Strides& get_window_dilation_strides_backward() const
            {
                return m_window_dilation_strides_backward;
            }
            /// \return The padding-below sizes (possibly negative) for the backward prop.
            const ngraph::CoordinateDiff& get_padding_below_backward() const
            {
                return m_padding_below_backward;
            }
            /// \return The padding-above sizes (possibly negative) for the backward prop.
            const ngraph::CoordinateDiff& get_padding_above_backward() const
            {
                return m_padding_above_backward;
            }
            /// \return The input data dilation strides for the backward prop.
            const ngraph::Strides& get_data_dilation_strides_backward() const
            {
                return m_data_dilation_strides_backward;
            }

        protected:
            ngraph::Shape m_data_batch_shape;
            ngraph::Strides m_window_movement_strides_forward;
            ngraph::Strides m_window_dilation_strides_forward;
            ngraph::CoordinateDiff m_padding_below_forward;
            ngraph::CoordinateDiff m_padding_above_forward;
            ngraph::Strides m_data_dilation_strides_forward;

            ngraph::Strides m_window_movement_strides_backward;
            ngraph::Strides m_window_dilation_strides_backward;
            ngraph::CoordinateDiff m_padding_below_backward;
            ngraph::CoordinateDiff m_padding_above_backward;
            ngraph::Strides m_data_dilation_strides_backward;
        };

        /// \brief Filters backprop for batched convolution operation.
        class ConvolutionBackpropFilters : public Op
        {
        public:
            /// \brief Constructs a batched-convolution filter-backprop operation.
            ///
            /// \param filters_shape The shape of the filters from forward-prop.
            /// \param window_movement_strides_forward The window movement strides from forward-prop.
            /// \param window_dilation_strides_forward The window dilation strides from forward-prop.
            /// \param padding_below_forward The padding-below sizes from forward-prop.
            /// \param padding_above_forward The padding-above sizes from forward-prop.
            /// \param data_dilation_strides_forward The data dilation strides from forward-prop.
            ConvolutionBackpropFilters(const ngraph::Shape& filters_shape,
                                       const ngraph::Strides& window_movement_strides_forward,
                                       const ngraph::Strides& window_dilation_strides_forward,
                                       const ngraph::CoordinateDiff& padding_below_forward,
                                       const ngraph::CoordinateDiff& padding_above_forward,
                                       const ngraph::Strides& data_dilation_strides_forward);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            /// \return The filters tensor shape.
            const ngraph::Shape& get_filters_shape() const { return m_filters_shape; }
            /// \return The window movement strides from the forward prop.
            const ngraph::Strides& get_window_movement_strides_forward() const
            {
                return m_window_movement_strides_forward;
            }
            /// \return The window dilation strides from the forward prop.
            const ngraph::Strides& get_window_dilation_strides_forward() const
            {
                return m_window_dilation_strides_forward;
            }
            /// \return The padding-below sizes (possibly negative) from the forward prop.
            const ngraph::CoordinateDiff& get_padding_below_forward() const
            {
                return m_padding_below_forward;
            }
            /// \return The padding-above sizes (possibly negative) from the forward prop.
            const ngraph::CoordinateDiff& get_padding_above_forward() const
            {
                return m_padding_above_forward;
            }
            /// \return The data dilation strides from the forward prop.
            const ngraph::Strides& get_data_dilation_strides_forward() const
            {
                return m_data_dilation_strides_forward;
            }

            /// \return The window movement strides for the backward prop.
            const ngraph::Strides& get_window_movement_strides_backward() const
            {
                return m_window_movement_strides_backward;
            }
            /// \return The window dilation strides for the backward prop.
            const ngraph::Strides& get_window_dilation_strides_backward() const
            {
                return m_window_dilation_strides_backward;
            }
            /// \return The padding-below sizes (possibly negative) for the backward prop.
            const ngraph::CoordinateDiff& get_padding_below_backward() const
            {
                return m_padding_below_backward;
            }
            /// \return The padding-above sizes (possibly negative) for the backward prop.
            const ngraph::CoordinateDiff& get_padding_above_backward() const
            {
                return m_padding_above_backward;
            }
            /// \return The data dilation strides for the backward prop.
            const ngraph::Strides& get_data_dilation_strides_backward() const
            {
                return m_data_dilation_strides_backward;
            }

        protected:
            ngraph::Shape m_filters_shape;
            ngraph::Strides m_window_movement_strides_forward;
            ngraph::Strides m_window_dilation_strides_forward;
            ngraph::CoordinateDiff m_padding_below_forward;
            ngraph::CoordinateDiff m_padding_above_forward;
            ngraph::Strides m_data_dilation_strides_forward;

            ngraph::Strides m_window_movement_strides_backward;
            ngraph::Strides m_window_dilation_strides_backward;
            ngraph::CoordinateDiff m_padding_below_backward;
            ngraph::CoordinateDiff m_padding_above_backward;
            ngraph::Strides m_data_dilation_strides_backward;
        };
    }
}
