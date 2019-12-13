// Microsoft (c) 2019, NNFusion Team

#include "max_pool.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/operators/util/validation_util.hpp"

using namespace std;
using namespace nnfusion::op;

MaxPool::MaxPool(const ngraph::Shape& window_shape,
                 const ngraph::Strides& window_movement_strides,
                 const ngraph::Shape& padding_below,
                 const ngraph::Shape& padding_above)
    : Op("MaxPool")
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
{
}

MaxPool::MaxPool(const ngraph::Shape& window_shape, const ngraph::Strides& window_movement_strides)
    : MaxPool(window_shape, window_movement_strides, ngraph::Shape(), ngraph::Shape())
{
}

MaxPool::MaxPool(const ngraph::Shape& window_shape)
    : MaxPool(window_shape, ngraph::Strides(), ngraph::Shape(), ngraph::Shape())
{
}

void MaxPool::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    if (0 == m_window_movement_strides.size())
    {
        m_window_movement_strides = ngraph::Strides(m_window_shape.size(), 1);
    }

    if (0 == m_padding_below.size())
    {
        m_padding_below = ngraph::Shape(m_window_shape.size(), 0);
    }

    if (0 == m_padding_above.size())
    {
        m_padding_above = ngraph::Shape(m_window_shape.size(), 0);
    }

    const ngraph::PartialShape& arg_shape = gnode->get_input_partial_shape(0);

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    ngraph::CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    ngraph::CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    gnode->set_output_type_and_shape(0,
                                     gnode->get_input_element_type(0),
                                     infer_batched_pooling_forward(this,
                                                                   arg_shape,
                                                                   padding_below,
                                                                   padding_above,
                                                                   m_window_shape,
                                                                   m_window_movement_strides,
                                                                   true));
}

MaxPoolBackprop::MaxPoolBackprop(const ngraph::Shape& window_shape,
                                 const ngraph::Strides& window_movement_strides,
                                 const ngraph::Shape& padding_below,
                                 const ngraph::Shape& padding_above,
                                 const shared_ptr<MaxPool>& forward_op)
    : Op("MaxPoolBackprop")
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_forward_op(forward_op)
{
}

void MaxPoolBackprop::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto forward_arg_et = gnode->get_input_element_type(0);
    auto delta_et = gnode->get_input_element_type(1);
    ngraph::element::Type result_et;

    OP_VALIDATION(this, ngraph::element::Type::merge(result_et, forward_arg_et, delta_et))
        << "Element types for forward argument (" << forward_arg_et << ") and delta (" << delta_et
        << ") do not match.";

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    ngraph::CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    ngraph::CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    const ngraph::PartialShape& forward_arg_shape = gnode->get_input_partial_shape(0);

    ngraph::PartialShape forward_result_shape =
        infer_batched_pooling_forward(this,
                                      forward_arg_shape,
                                      padding_below,
                                      padding_above,
                                      m_window_shape,
                                      m_window_movement_strides,
                                      true);

    const ngraph::PartialShape& delta_shape = gnode->get_input_partial_shape(1);

    OP_VALIDATION(this, forward_result_shape.compatible(delta_shape))
        << "Inferred forward output shape does not match delta shape (inferred forward output "
        << "shape: " << forward_result_shape << ", delta shape: " << delta_shape << ").";

    // TODO(amprocte): We may technically be able to infer some extra information from
    // forward_result_shape that was not present in the forward arg shape---namely batch size and
    // channel count. Merge that info in.
    gnode->set_output_type_and_shape(0, forward_arg_et, forward_arg_shape);
}

shared_ptr<MaxPool> MaxPoolBackprop::get_forward_op() const
{
    return m_forward_op.lock();
}
