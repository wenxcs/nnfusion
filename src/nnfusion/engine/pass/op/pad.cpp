// Microsoft (c) 2019, Wenxiang
#include "pad.h"

using namespace nnfusion::ir;

Pad::Pad(shared_ptr<Node> node)
    : Operator(node)
{
    auto pad = static_pointer_cast<ngraph::op::Pad>(node);
    input_shape = ngraph::Shape(args[0].get_shape());
    output_shape = ngraph::Shape(out[0].get_shape());
    padding_below = ngraph::Shape(pad->get_padding_below());
    padding_above = ngraph::Shape(pad->get_padding_above());
    padding_interior = ngraph::Shape(pad->get_padding_interior());
    input_type = args[0].get_element_type().c_type_string();
    output_type = out[0].get_element_type().c_type_string();

    rank = static_cast<uint32_t>(input_shape.size());

    pad_below = ngraph::NVShape(input_shape.size(), 0);
    pad_interior = ngraph::NVShape(input_shape.size(), 1);

    int64_t i = padding_below.size() - 1;
    int64_t j = input_shape.size() - 1;
    for (; i >= 0; i--, j--)
    {
        pad_below[j] = padding_below[i];
        pad_interior[j] = padding_interior[i];
    }

    input_strides = row_major_strides(input_shape);
    output_strides = row_major_strides(output_shape);
}

Operator_p Pad::translate(shared_ptr<Node> node)
{
    create_ptr(Pad, inter_op, node);
    return inter_op;
}