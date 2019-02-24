// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Anyop : public Operator
        {
        public:
            vector<string> dtypes;
            vector<string> dsizes;

        public:
            Anyop(shared_ptr<Node> node)
                : Operator(node)
            {
                for (auto& arg : args)
                {
                    this->dtypes.push_back(arg.get_type());
                    this->dsizes.push_back(to_string(shape_size(arg.get_tensor().get_shape())));
                }

                for (auto& ou : out)
                {
                    this->dtypes.push_back(ou.get_type());
                    this->dsizes.push_back(to_string(shape_size(ou.get_tensor().get_shape())));
                }
            }

            static Operator_p translate(shared_ptr<Node> node)
            {
                create_ptr(Anyop, inter_op, node);
                return inter_op;
            }
        };

        using Anyop_p = shared_ptr<Anyop>;
    }
}