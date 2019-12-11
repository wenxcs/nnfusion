// Microsoft (c) 2019, NNFusion Team

#include <memory>

#include "index_reduction.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

IndexReduction::IndexReduction(const std::string& node_type,
                               size_t axis,
                               const ngraph::element::Type& index_element_type)
    : Op(node_type)
    , m_axis(axis)
    , m_index_element_type(index_element_type)
{
}

void IndexReduction::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    const ngraph::PartialShape& arg_shape = gnode->get_input_partial_shape(0);
    ngraph::Rank rank = arg_shape.rank();

    OP_VALIDATION(this, rank.is_dynamic() || size_t(rank) >= 1) << "Argument rank is zero.";
    OP_VALIDATION(this, rank.is_dynamic() || m_axis < size_t(rank))
        << "Reduction axis (" << m_axis << ") is not less than argument rank (" << rank << ").";
    OP_VALIDATION(this,
                  m_index_element_type == ngraph::element::i32 ||
                      m_index_element_type == ngraph::element::i64)
        << "Index element is neither i64 or i32.";

    ngraph::PartialShape output_shape{ngraph::PartialShape::dynamic()};

    if (!rank.is_dynamic())
    {
        std::vector<ngraph::Dimension> output_dims(size_t(rank) - 1);
        size_t j = 0;

        for (size_t i = 0; i < size_t(rank) - 1; i++)
        {
            if (j == m_axis)
            {
                j++;
            }
            output_dims[i] = arg_shape[j++];
        }

        output_shape = ngraph::PartialShape(output_dims);
    }

    set_output_type_and_shape(gnode, 0, m_index_element_type, output_shape);
}
