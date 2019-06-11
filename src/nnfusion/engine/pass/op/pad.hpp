// Microsoft (c) 2019, Wenxiang
#pragma once

#include "nnfusion/engine/op.hpp"
#include "external/nvshape.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Pad : public Operator
        {
        public:
            ngraph::Shape input_shape, output_shape, padding_above, padding_below, padding_interior;
            uint32_t rank;
            ngraph::NVShape input_strides, output_strides, pad_below, pad_interior;
            string input_type, output_type;

        public:
            Pad(shared_ptr<Node> node);

            static Operator_p translate(shared_ptr<Node> node);
        };

        using Pad_p = shared_ptr<Pad>;
    }
}