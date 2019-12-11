// Microsoft (c) 2019, NNFusion Team

#include <memory>

#include "ngraph/axis_vector.hpp"
#include "ngraph/shape.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "topk.hpp"

using namespace std;
using namespace nnfusion::op;

TopK::TopK(size_t top_k_axis,
           const ngraph::element::Type& index_element_type,
           size_t k,
           bool compute_max)
    : Op("TopK")
    , m_top_k_axis(top_k_axis)
    , m_index_element_type(index_element_type)
    , m_k(k)
    , m_compute_max(compute_max)
{
}

void TopK::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto input_shape = gnode->get_input_partial_shape(0);
    ngraph::Rank input_rank = input_shape.rank();
    ngraph::element::Type input_element_type = gnode->get_input_element_type(0);

    OP_VALIDATION(this, !m_index_element_type.is_dynamic())
        << "Argument element type must not be dynamic.";

    OP_VALIDATION(this,
                  m_index_element_type == ngraph::element::i32 ||
                      m_index_element_type == ngraph::element::i64)
        << "Argument element type must be i64 or i32 (got " << m_index_element_type << ").";

    OP_VALIDATION(this, input_rank.is_dynamic() || static_cast<size_t>(input_rank) > 0)
        << "Argument rank must be greater than 0.";

    OP_VALIDATION(this, input_rank.is_dynamic() || m_top_k_axis < static_cast<size_t>(input_rank))
        << "TopK axis (" << m_top_k_axis << ") is out of bounds.";

    OP_VALIDATION(this,
                  input_rank.is_dynamic() || input_shape[m_top_k_axis].is_dynamic() ||
                      m_k <= static_cast<size_t>(input_shape[m_top_k_axis]))
        << "K (" << m_k << ") exceeds the dimension ("
        << (input_rank.is_static() ? input_shape[m_top_k_axis] : 0) << ") of the TopK axis (axis "
        << m_top_k_axis << ").";

    ngraph::PartialShape output_shape{input_shape};

    if (input_rank.is_static())
    {
        if (m_k != 0)
        {
            output_shape[m_top_k_axis] = m_k;
        }
        else if (input_shape[m_top_k_axis].is_static())
        {
            m_k = static_cast<size_t>(input_shape[m_top_k_axis]);
        }
    }

    gnode->set_output_size(2);
    set_output_type_and_shape(gnode, 0, m_index_element_type, output_shape);
    set_output_type_and_shape(gnode, 1, input_element_type, output_shape);
}