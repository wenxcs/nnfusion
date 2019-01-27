// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Noop : public Operator
        {
        public:
            Noop(shared_ptr<Node> node)
                : Operator(node)
            {
            }

            static Operator_p translate(shared_ptr<Node> node)
            {
                create_ptr(Noop, inter_op, node);
                return inter_op;
            }
        };

        using Noop_p = shared_ptr<Noop>;
    }
}