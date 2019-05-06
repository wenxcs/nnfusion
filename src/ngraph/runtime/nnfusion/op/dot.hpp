// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Dot : public Operator
        {
        public:
            size_t reduction_axes;
            ngraph::Shape arg0_shape, arg1_shape;
            ngraph::Shape out_shape;
            ngraph::element::Type dtype;

        public:
            Dot(shared_ptr<Node> node)
                : Operator(node)
            {
                auto dot = static_pointer_cast<ngraph::op::Dot>(node);
                enforce_not_nullptr(dot) << "Invalid input.";
                reduction_axes = dot->get_reduction_axes_count();
                arg0_shape = ngraph::Shape(args[0].get_shape());
                arg1_shape = ngraph::Shape(args[1].get_shape());
                out_shape = ngraph::Shape(out[0].get_shape());
                dtype = ngraph::element::Type(out[0].get_element_type());
            }

            static Operator_p translate(shared_ptr<Node> node)
            {
                create_ptr(Dot, inter_op, node);
                return inter_op;
            }
        };

        using Dot_p = shared_ptr<Dot>;
    }
}