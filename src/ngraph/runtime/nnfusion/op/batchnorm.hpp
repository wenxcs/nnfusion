// Microsoft (c) 2019, Wenxiang
/**
 * \class BatchNorm
 * \brief Intermediate representation for Average Pooling Operator
 * \note the BatchNorm maybe codegened to different op by the dimenstion(1d, 2d, 3d...)
 * \author wenxh
 */

#pragma once

#include "../core/op.hpp"

namespace nnfusion
{
    namespace ir
    {
        class BatchNorm : public Operator
        {
        public:
            Shape tensor_shape, param_shape;
            double epsilon;
            bool global_stats, save_states;
            string dtype;
            string direction = "CUDNNEmitter::Prop::Inference";

        public:
            /// Create an instance of BatchNorm.
            BatchNorm(shared_ptr<Node> node)
                : Operator(node)
            {
                // ngraph::op::BatchNormInferece <-> nnfusion::ir::BatchNorm
                auto bn_op = static_pointer_cast<op::BatchNormInference>(node);
                dtype = out[0].get_type();
                tensor_shape = args[2].get_shape();
                param_shape = args[0].get_shape();
                epsilon = bn_op->get_eps_value();

                /*Future purpose*/
                global_stats = false;
                save_states = false;
            }

            /// Translate the operator from ngraph::node, deal with boundry cases in this function.
            static Operator_p translate(shared_ptr<Node> node)
            {
                create_ptr(BatchNorm, inter_op, node);
                auto& input_shape = inter_op->args[0].get_shape();
                // Sanity check: Currently we only support 2D/3D symetric padding
                return inter_op;
            }
        };

        /// Alias for pointer to BatchNorm Object.
        using BatchNorm_p = shared_ptr<BatchNorm>;
    }
}