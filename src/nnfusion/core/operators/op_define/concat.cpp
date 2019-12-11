// Microsoft (c) 2019, NNFusion Team

#include <cassert>
#include <memory>

#include "concat.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

Concat::Concat(size_t concatenation_axis)
    : Op("Concat")
    , m_concatenation_axis(concatenation_axis)
{
}

void Concat::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    OP_VALIDATION(this, gnode->get_input_size() >= 1) << "At least one argument required.";

    ngraph::PartialShape inputs_shape_scheme{ngraph::PartialShape::dynamic()};
    ngraph::element::Type inputs_et{ngraph::element::dynamic};
    ngraph::Dimension concatenation_axis_output_dim{0};

    for (auto i = 0; i < gnode->get_input_size(); i++)
    {
        ngraph::PartialShape this_input_shape = gnode->get_input_partial_shape(i);
        ngraph::Dimension this_input_rank = this_input_shape.rank();
        if (this_input_rank.is_static())
        {
            OP_VALIDATION(this, m_concatenation_axis < size_t(this_input_rank))
                << "Concatenation axis (" << m_concatenation_axis << ") is out of bounds for "
                << "argument " << i << ", which has shape " << this_input_shape << ".";

            concatenation_axis_output_dim += this_input_shape[m_concatenation_axis];
            this_input_shape[m_concatenation_axis] = ngraph::Dimension::dynamic();

            OP_VALIDATION(this,
                          ngraph::PartialShape::merge_into(inputs_shape_scheme, this_input_shape))
                << "Argument shapes are inconsistent; they must have the same rank, and must have "
                << "equal dimension everywhere except on the concatenation axis (axis "
                << m_concatenation_axis << ").";

            OP_VALIDATION(this,
                          ngraph::element::Type::merge(
                              inputs_et, inputs_et, gnode->get_input_element_type(i)))
                << "Argument element types are inconsistent.";
        }
        else
        {
            concatenation_axis_output_dim += ngraph::Dimension::dynamic();
        }
    }

    ngraph::PartialShape concatenated_shape = inputs_shape_scheme;

    if (concatenated_shape.rank().is_static())
    {
        concatenated_shape[m_concatenation_axis] = concatenation_axis_output_dim;
    }

    set_output_type_and_shape(gnode, 0, inputs_et, concatenated_shape);
}
