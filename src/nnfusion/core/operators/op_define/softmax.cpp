// Microsoft (c) 2019, NNFusion Team

#include "softmax.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace nnfusion::op;

Softmax::Softmax(const nnfusion::AxisSet& axes, bool in_log_space)
    : ElementwiseArithmetic("Softmax")
    , m_axes(axes)
    , m_in_log_space(in_log_space)
{
}

void Softmax::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    ElementwiseArithmetic::validate_and_infer_types(gnode);

    auto shape = gnode->get_output_shape(0);

    NNFUSION_CHECK(m_axes.empty() ||
                   (m_axes.size() == 1 && *std::begin(m_axes) == shape.size() - 1))
        << "softmax only support computing on the last dim";
    for (auto axis : m_axes)
    {
        OP_VALIDATION(this, axis < shape.size()) << "Reduction axis (" << axis
                                                 << ") is out of bounds (argument shape: " << shape
                                                 << ").";
    }

    // empty axes == all axes
    if (m_axes.size() == 0)
    {
        for (size_t i = 0; i < shape.size(); ++i)
        {
            m_axes.insert(i);
        }
    }
}

SoftmaxGrad::SoftmaxGrad(const nnfusion::AxisSet& axes, bool in_log_space)
    : Op("SoftmaxGrad")
    , m_axes(axes)
    , m_in_log_space(in_log_space)
{
}

void SoftmaxGrad::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    NNFUSION_CHECK(gnode->get_input_size() == 2);
    NNFUSION_CHECK(gnode->get_input_shape(0) == gnode->get_input_shape(1));

    gnode->set_output_type_and_shape(
        0, gnode->get_input_element_type(0), gnode->get_input_shape(0));

    auto shape = gnode->get_output_shape(0);

    NNFUSION_CHECK(m_axes.empty() ||
                   (m_axes.size() == 1 && *std::begin(m_axes) == shape.size() - 1))
        << "softmax_grad only support computing on the last dim";

    for (auto axis : m_axes)
    {
        OP_VALIDATION(this, axis < shape.size()) << "Reduction axis (" << axis
                                                 << ") is out of bounds (argument shape: " << shape
                                                 << ").";
    }

    // empty axes == all axes
    if (m_axes.size() == 0)
    {
        for (size_t i = 0; i < shape.size(); ++i)
        {
            m_axes.insert(i);
        }
    }
}
