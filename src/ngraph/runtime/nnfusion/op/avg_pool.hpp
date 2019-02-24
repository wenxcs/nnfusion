// Microsoft (c) 2019, Wenxiang
/**
 * \class AvgPool
 * \brief Intermediate representation for Average Pooling Operator
 * \note the AvgPool maybe codegened to different op by the dimenstion(1d, 2d, 3d...)
 * \author wenxh
 */

#pragma once

#include "../core/op.hpp"

namespace nnfusion
{
    namespace ir
    {
        class AvgPool : public Operator
        {
        public:
            Shape input_shape, result_shape, padding_below, padding_above, window_shape,
                window_stride;
            bool include_pad;

        public:
            /// Create an instance of AvgPool.
            AvgPool(shared_ptr<Node> node)
                : Operator(node)
            {
                auto avg_pool = static_pointer_cast<op::AvgPool>(node);
                input_shape = args[0].get_shape();
                result_shape = out[0].get_shape();
                window_shape = avg_pool->get_window_shape();
                window_stride = avg_pool->get_window_movement_strides();
                padding_below = avg_pool->get_padding_below();
                padding_above = avg_pool->get_padding_above();
                include_pad = avg_pool->get_include_padding_in_avg_computation();
            }

            /// Translate the operator from ngraph::node, deal with boundry cases in this function.
            static Operator_p translate(shared_ptr<Node> node)
            {
                create_ptr(AvgPool, inter_op, node);
                auto& input_shape = inter_op->args[0].get_shape();
                // Sanity check: Currently we only support 2D/3D symetric padding
                return inter_op;
            }
        };

        /// Alias for pointer to AvgPool Object.
        using AvgPool_p = shared_ptr<AvgPool>;
    }
}