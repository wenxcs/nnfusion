// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"
#include "util/nvshape.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Slice : public Operator
        {
        public:
            NVShape input_shape, lower_bounds, slice_strides, output_shape;

        public:
            Slice(shared_ptr<Node> node)
                : Operator(node)
            {
                auto slice = static_pointer_cast<op::Slice>(node);
                input_shape = args[0].get_shape();
                output_shape = out[0].get_shape();
                lower_bounds = slice->get_lower_bounds();
                slice_strides = slice->get_strides();
            }

            static Operator_p translate(shared_ptr<Node> node)
            {
                create_ptr(Slice, inter_op, node);
                return inter_op;
            }
        };

        using Slice_p = shared_ptr<Slice>;
    }
}