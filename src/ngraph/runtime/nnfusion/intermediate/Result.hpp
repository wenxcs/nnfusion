// Microsoft (c) 2019, Wenxiang
#pragma once

#include "ngraph/runtime/nnfusion/intermediate/Noop.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_op.hpp"

using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            namespace intermediate
            {
                class Result : public IntermediateOP
                {
                public:
                    Result(shared_ptr<Node> node)
                        : IntermediateOP(node)
                    {
                    }

                    static std::shared_ptr<IntermediateOP> translate(shared_ptr<Node> node)
                    {
                        std::shared_ptr<Result> inter_op(new Result(node));

                        if (inter_op->args[0].get_name() == inter_op->out[0].get_name())
                        {
                            shared_ptr<Noop> notrans(new Noop(node));
                            NGRAPH_DEBUG << "Skipping translation for " << node->get_name() << "\n";
                            return notrans;
                        }

                        return inter_op;
                    }
                };
            }
        }
    }
}