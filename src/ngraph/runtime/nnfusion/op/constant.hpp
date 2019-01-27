// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"
#include "noop.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Constant : public Operator
        {
        public:
            const void* data_ptr;
            size_t data_size;
            string dtype;

        public:
            Constant(shared_ptr<op::Constant> node)
                : data_ptr(nullptr)
                , data_size(0)
                , Operator(node)
            {
                data_ptr = node->get_data_ptr();
                data_size = node->get_data_size();
                assert_bool(out.size() == 1) << "Constant only has one output.";
                dtype = out[0].get_type();
            }

            static Operator_p translate(shared_ptr<Node> node)
            {
                auto cons_ptr = dynamic_pointer_cast<op::Constant>(node);
                assert_nullptr(cons_ptr)
                    << "Only support translation for Constant node in this function";
                create_ptr(Constant, inter_op, cons_ptr);
                NGRAPH_DEBUG << "Translated constant:" << node->get_name() << endl;
                return inter_op;
            }
        };

        using Constant_p = shared_ptr<Constant>;
    }
}