// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "nnfusion/core/operators/op.hpp"

namespace nnfusion
{
    namespace op
    {
        ngraph::PartialShape
            infer_windowed_reduction_output_shape(const Op* op,
                                                  const ngraph::PartialShape& data_shape,
                                                  const ngraph::Strides& data_dilation,
                                                  const ngraph::CoordinateDiff& data_padding_below,
                                                  const ngraph::CoordinateDiff& data_padding_above,
                                                  const ngraph::PartialShape& window_shape,
                                                  const ngraph::Strides& window_strides,
                                                  const ngraph::Strides& window_dilation,
                                                  bool is_window_all_in_padding_allowed);

        std::tuple<ngraph::element::Type, ngraph::PartialShape>
            infer_convolution_forward(const Op* op,
                                      ngraph::element::Type et_batch,
                                      ngraph::element::Type et_filters,
                                      const ngraph::PartialShape& data_batch_shape,
                                      const ngraph::Strides& data_dilation,
                                      const ngraph::CoordinateDiff& data_padding_below,
                                      const ngraph::CoordinateDiff& data_padding_above,
                                      const ngraph::PartialShape& filters_shape,
                                      const ngraph::Strides& filter_strides,
                                      const ngraph::Strides& filter_dilation);
        ngraph::PartialShape
            infer_batched_pooling_forward(const Op* op,
                                          const ngraph::PartialShape& data_batch_shape,
                                          const ngraph::CoordinateDiff& data_padding_below,
                                          const ngraph::CoordinateDiff& data_padding_above,
                                          const ngraph::PartialShape& window_shape,
                                          const ngraph::Strides& window_strides,
                                          bool is_window_all_in_padding_allowed);

        std::tuple<ngraph::element::Type, ngraph::PartialShape, ngraph::PartialShape>
            infer_batch_norm_forward(const Op* op,
                                     ngraph::element::Type input_element_type,
                                     ngraph::element::Type gamma_element_type,
                                     ngraph::element::Type beta_element_type,
                                     ngraph::element::Type mean_element_type,
                                     ngraph::element::Type variance_element_type,
                                     const ngraph::PartialShape& input_shape,
                                     const ngraph::PartialShape& gamma_shape,
                                     const ngraph::PartialShape& beta_shape,
                                     const ngraph::PartialShape& mean_shape,
                                     const ngraph::PartialShape& variance_shape);

        std::tuple<ngraph::element::Type, ngraph::PartialShape, ngraph::PartialShape>
            infer_batch_norm_forward(const Op* op,
                                     ngraph::element::Type input_element_type,
                                     ngraph::element::Type gamma_element_type,
                                     ngraph::element::Type beta_element_type,
                                     const ngraph::PartialShape& input_shape,
                                     const ngraph::PartialShape& gamma_shape,
                                     const ngraph::PartialShape& beta_shape);
    }
}
