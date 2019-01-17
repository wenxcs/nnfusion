// Microsoft (c) 2019, Wenxiang
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
                class NoTrans : public IntermediateOP
                {
                public:
                    static std::shared_ptr<IntermediateOP> translate(TRANS_ARGS)
                    {
                        std::shared_ptr<NoTrans> inter_op(new NoTrans());
                        inter_op->n = node;
                        inter_op->args = args;
                        inter_op->out = out;

                        return inter_op;
                    }
                };
            }
        }
    }
}