// Microsoft (c) 2019, NNFusion Team

#include "arithmetic_reduction.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

ArithmeticReduction::ArithmeticReduction(const std::string& node_type,
                                         const ngraph::AxisSet& reduction_axes)
    : Op(node_type)
    , m_reduction_axes(reduction_axes)
{
}

void ArithmeticReduction::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto input_shape = gnode->get_input_partial_shape(0);
    auto input_rank = input_shape.rank();

    ngraph::PartialShape result_shape{ngraph::PartialShape::dynamic()};

    if (input_rank.is_static())
    {
        std::vector<ngraph::Dimension> dims;

        for (auto axis : m_reduction_axes)
        {
            OP_VALIDATION(this, axis < size_t(input_rank))
                << "Reduction axis (" << axis << ") is out of bounds "
                << "(argument shape: " << input_shape << ", reduction axes: " << m_reduction_axes
                << ")";
        }

        for (size_t i = 0; i < size_t(input_rank); i++)
        {
            if (m_reduction_axes.count(i) == 0)
            {
                dims.push_back(input_shape[i]);
            }
        }

        result_shape = ngraph::PartialShape(dims);
    }

    gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), result_shape);
}
