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
                    Result(shared_ptr<Node> node);

                    static std::shared_ptr<IntermediateOP> translate(shared_ptr<Node> node);
                };
            }
        }
    }
}