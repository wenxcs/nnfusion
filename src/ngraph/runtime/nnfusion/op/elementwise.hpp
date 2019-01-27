// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"
#include "noop.hpp"

namespace nnfusion
{
    namespace ir
    {
        template <class T>
        class Elementwise : public Operator
        {
        public:
            vector<string> dtypes;

        public:
            Elementwise(shared_ptr<Node> node)
                : Operator(node)
            {
                assert_bool(out.size() == 1)
                    << "Multi-output elementwise ops are not currently supported.";

                for (auto& arg : args)
                {
                    this->dtypes.push_back(arg.get_type());
                }
                this->dtypes.push_back(out[0].get_type());
            }

            static Operator_p translate(shared_ptr<Node> node)
            {
                create_ptr(Elementwise, inter_op, node);

                if (inter_op->out[0].get_size() == 0)
                {
                    create_ptr(Noop, notrans, node);
                    return notrans;
                }

                NGRAPH_DEBUG << "Translated " << node->get_name() << endl;
                return inter_op;
            }
        };
    }
}