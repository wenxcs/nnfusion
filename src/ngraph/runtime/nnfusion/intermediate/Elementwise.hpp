// Microsoft (c) 2019, Wenxiang
#pragma once

#include "ngraph/runtime/nnfusion/intermediate/Noop.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_op.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            namespace intermediate
            {
                template <class T>
                class Elementwise : public IntermediateOP
                {
                public:
                    vector<string> dtypes;

                public:
                    Elementwise(shared_ptr<Node> node)
                        : IntermediateOP(node)
                    {
                        assert_bool(out.size() == 1)
                            << "Multi-output elementwise ops are not currently supported.";

                        for (auto& arg : args)
                        {
                            this->dtypes.push_back(arg.get_type());
                        }
                        this->dtypes.push_back(out[0].get_type());
                    }

                    static std::shared_ptr<IntermediateOP> translate(shared_ptr<Node> node)
                    {
                        std::shared_ptr<Elementwise> inter_op(new Elementwise(node));

                        if (inter_op->out[0].get_size() == 0)
                        {
                            shared_ptr<Noop> notrans(new Noop(node));
                            return notrans;
                        }

                        NGRAPH_DEBUG << "Translated " << node->get_name() << endl;
                        return inter_op;
                    }
                };
            }
        }
    }
}