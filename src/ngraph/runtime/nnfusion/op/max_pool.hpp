// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"

namespace nnfusion
{
    namespace ir
    {
        class MaxPool : public Operator
        {
        public:
            ngraph::Shape input_shape;
            ngraph::Shape output_shape;
            ngraph::Shape padding_below;
            ngraph::Shape padding_above;
            ngraph::Strides window_stride;
            ngraph::Shape window_shape;

        public:
            MaxPool(shared_ptr<Node> node)
                : Operator(node)
            {
                //<Todo> a layout check
                // assumes NC{d1,d2,...} format
                auto max_pool = static_pointer_cast<ngraph::op::MaxPool>(node);

                assert_bool(args.size() == 1 && out.size() == 1)
                    << "Maxpool only has one input & one output.";

                input_shape = ngraph::Shape(args[0].get_shape());
                output_shape = ngraph::Shape(out[0].get_shape());
                padding_below = max_pool->get_padding_below();
                padding_above = max_pool->get_padding_above();

                //<todo:wenxh> support it in the future with Graph
                assert_bool(padding_below == padding_above)
                    << "Asymmetric padding for Maxpooling is not supported by now.";

                window_stride = ngraph::Strides(max_pool->get_window_movement_strides());
                window_shape = ngraph::Shape(max_pool->get_window_shape());

                assert_bool(input_shape.size() >= 3 && input_shape.size() <= 5);
            }

            static Operator_p translate(shared_ptr<Node> node)
            {
                create_ptr(MaxPool, inter_op, node);
                auto& input_shape = inter_op->args[0].get_shape();
                assert_bool(input_shape.size() >= 3)
                    << "MaxPool operation requested for a tensor of less than 3 dimensions. "
                       "Tensors should have at least one spatial dimension, dim(NC{d1...dN}) "
                       "<= 3";
                assert_bool(input_shape.size() <= 5)
                    << "Pooling currently only supports up to 3 spatial dimensions.";
                return inter_op;
            }
        };

        using MaxPool_p = shared_ptr<MaxPool>;
    }
}