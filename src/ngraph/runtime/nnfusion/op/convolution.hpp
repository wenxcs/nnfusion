// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Convolution : public Operator
        {
        public:
            shared_ptr<ngraph::op::Convolution> node;
            ngraph::Shape input_shape, filter_shape, output_shape;
            ngraph::Strides window_dilation_strides, window_movement_strides, data_dilation_strides;
            ngraph::CoordinateDiff padding_below_diff, padding_above_diff;
            string dtype;

        public:
            Convolution(shared_ptr<Node> node)
                : Operator(node)
            {
                this->node = static_pointer_cast<ngraph::op::Convolution>(node);
                assert_nullptr(this->node) << "Input ngraph::Node is invalid.";

                input_shape = args[0].get_shape();
                filter_shape = args[1].get_shape();
                output_shape = out[0].get_shape();
                window_dilation_strides = this->node->get_window_dilation_strides();
                window_movement_strides = this->node->get_window_movement_strides();
                data_dilation_strides = this->node->get_data_dilation_strides();
                padding_below_diff = this->node->get_padding_below();
                padding_above_diff = this->node->get_padding_above();
                dtype = out[0].get_element_type().c_type_string();
            }

            static Operator_p translate(shared_ptr<Node> node)
            {
                create_ptr(Convolution, inter_op, node);
                return inter_op;
            }
        };

        using Convolution_p = shared_ptr<Convolution>;
    }
}