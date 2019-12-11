// Microsoft (c) 2019, NNFusion Team

#include "avg_pool.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/operators/util/validation_util.hpp"

using namespace std;
using namespace nnfusion::op;

AvgPool::AvgPool(const ngraph::Shape& window_shape,
                 const ngraph::Strides& window_movement_strides,
                 const ngraph::Shape& padding_below,
                 const ngraph::Shape& padding_above,
                 bool include_padding_in_avg_computation)
    : Op("AvgPool")
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_include_padding_in_avg_computation(include_padding_in_avg_computation)
{
}

void AvgPool::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    if (0 == m_window_movement_strides.size())
    {
        m_window_movement_strides = Strides(m_window_shape.size(), 1);
    }

    if (0 == m_padding_below.size())
    {
        m_padding_below = Shape(m_window_shape.size(), 0);
    }

    if (0 == m_padding_above.size())
    {
        m_padding_above = Shape(m_window_shape.size(), 0);
    }

    const ngraph::PartialShape& arg_shape = gnode->get_input_partial_shape(0);

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    ngraph::CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    ngraph::CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    set_output_type_and_shape(gnode,
                              0,
                              gnode->get_input_element_type(0),
                              infer_batched_pooling_forward(this,
                                                            arg_shape,
                                                            padding_below,
                                                            padding_above,
                                                            m_window_shape,
                                                            m_window_movement_strides,
                                                            m_include_padding_in_avg_computation));
}

AvgPool::AvgPool(const Shape& window_shape, const Strides& window_movement_strides)
    : AvgPool(window_shape, window_movement_strides, ngraph::Shape(), ngraph::Shape(), false)
{
}

AvgPool::AvgPool(const Shape& window_shape)
    : AvgPool(window_shape, Strides(), Shape(), Shape(), false)
{
}

AvgPoolBackprop::AvgPoolBackprop(const ngraph::Shape& forward_arg_shape,
                                 const ngraph::Shape& window_shape,
                                 const ngraph::Strides& window_movement_strides,
                                 const ngraph::Shape& padding_below,
                                 const ngraph::Shape& padding_above,
                                 bool include_padding_in_avg_computation)
    : Op("AvgPoolBackprop")
    , m_forward_arg_shape(forward_arg_shape)
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_include_padding_in_avg_computation(include_padding_in_avg_computation)
{
}

void AvgPoolBackprop::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    ngraph::CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    ngraph::CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    ngraph::PartialShape forward_result_shape =
        infer_batched_pooling_forward(this,
                                      m_forward_arg_shape,
                                      padding_below,
                                      padding_above,
                                      m_window_shape,
                                      m_window_movement_strides,
                                      m_include_padding_in_avg_computation);

    const ngraph::PartialShape& delta_shape = gnode->get_input_shape(0);

    OP_VALIDATION(this, forward_result_shape.compatible(delta_shape))
        << "Inferred forward output shape does not match delta shape (inferred forward output "
        << "shape: " << forward_result_shape << ", delta shape: " << delta_shape << ").";

    // TODO(amprocte): Once m_forward_arg_shape is allowed to be dynamic, we may technically be
    // able to infer some extra information from forward_result_shape that was not present in the
    // forward arg shape---namely batch size and channel count. Merge that info in.
    set_output_type_and_shape(gnode, 0, gnode->get_input_element_type(0), m_forward_arg_shape);
}