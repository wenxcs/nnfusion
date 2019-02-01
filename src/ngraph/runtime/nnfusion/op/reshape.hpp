// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"
#include "util/nvshape.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Reshape : public Operator
        {
        public:
            ngraph::Shape arg_shape;
            size_t arg_rank;
            ngraph::Shape result_shape;
            ngraph::AxisVector input_order;
            shared_ptr<ngraph::op::Reshape> reshape;

        public:
            Reshape(shared_ptr<Node> node);

            bool isMemcpy();
            bool isNoop();

            static Operator_p translate(shared_ptr<Node> node);
        };

        using Reshape_p = shared_ptr<Reshape>;
    }
}