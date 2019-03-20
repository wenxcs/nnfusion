// Microsoft (c) 2019, Yuchao
#pragma once

#include "../core/op.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Softmax : public Operator
        {
        public:
            ngraph::AxisSet reduce_axis;
            shared_ptr<ngraph::op::Softmax> softmax;
            ngraph::Shape input_shape;

        public:
            Softmax(shared_ptr<Node> node)
                : Operator(node)
            {
                softmax = static_pointer_cast<ngraph::op::Softmax>(node);
                reduce_axis = softmax->get_axes();
                input_shape = args[0].get_shape();
            }

            static Operator_p translate(shared_ptr<Node> node)
            {
                create_ptr(Softmax, inter_op, node);
                return inter_op;
            }
        };

        using Softmax_p = shared_ptr<Softmax>;
    }
}