// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"
#include "reduce.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Sum : public Reduce
        {
        public:
            Sum(shared_ptr<Node> node)
                : Reduce(node)
            {
                auto sum = static_pointer_cast<ngraph::op::Sum>(node);
                assert_bool(args[0].get_size() != 0) << "Reset memory instead of Sum";
                assert_bool(args[0].get_size() != out[0].get_size()) << "Memcpy instead of Sum";
                auto axes_set = sum->get_reduction_axes();
                for (auto a : axes_set)
                {
                    reduce_axis.push_back(a);
                }
            }

            static Operator_p translate(shared_ptr<Node> node)
            {
                create_ptr(Sum, inter_op, node);
                return inter_op;
            }
        };

        using Sum_p = shared_ptr<Sum>;
    }
}