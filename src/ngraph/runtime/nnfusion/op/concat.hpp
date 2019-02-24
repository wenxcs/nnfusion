// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"
#include "util/nvshape.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Concat : public Operator
        {
        public:
            size_t axis;
            vector<NVShape> input_shapes;
            string dtype;
            Shape output_shape;

        public:
            Concat(shared_ptr<Node> node)
                : Operator(node)
            {
                auto concat = static_pointer_cast<ngraph::op::Concat>(node);
                this->axis = concat->get_concatenation_axis();
                for (auto arg : args)
                {
                    this->input_shapes.push_back(arg.get_shape());
                }
                this->dtype = out[0].get_type();
                this->output_shape = out[0].get_shape();
            }

            static Operator_p translate(shared_ptr<Node> node)
            {
                create_ptr(Concat, inter_op, node);
                return inter_op;
            }
        };

        using Concat_p = shared_ptr<Concat>;
    }
}