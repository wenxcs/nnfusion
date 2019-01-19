// Microsoft (c) 2019, Wenxiang
#pragma once

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
                class Noop : public IntermediateOP
                {
                public:
                    Noop(shared_ptr<Node> node)
                        : IntermediateOP(node)
                    {
                    }

                    static std::shared_ptr<IntermediateOP> translate(shared_ptr<Node> node)
                    {
                        std::shared_ptr<Noop> inter_op(new Noop(node));
                        return inter_op;
                    }
                };
            }
        }
    }
}