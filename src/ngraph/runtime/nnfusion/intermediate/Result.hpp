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
                class Result : public IntermediateOP
                {
                public:
                    static std::shared_ptr<IntermediateOP> translate(TRANS_ARGS)
                    {
                        std::shared_ptr<Result> inter_op(new Result());

                        if (args[0].get_name() == out[0].get_name())
                        {
                            std::cout << "// Skipping generation for " << node->get_name() << "\n";
                            return inter_op;
                        }

                        inter_op->n = node;
                        inter_op->args = args;
                        inter_op->out = out;

                        inter_op->isTranslated = true;
                        return inter_op;
                    }
                };
            }
        }
    }
}