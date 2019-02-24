// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Reduce : public Operator
        {
        public:
            ngraph::AxisVector reduce_axis;

        public:
            Reduce(shared_ptr<Node> node)
                : Operator(node)
            {
            }

            static Operator_p translate(shared_ptr<Node> node)
            {
                create_ptr(Reduce, inter_op, node);
                return inter_op;
            }
        };

        using Reduce_p = shared_ptr<Reduce>;
    }
}