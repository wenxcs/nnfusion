// Microsoft (c) 2019, Wenxiang
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
                class elementwise : public IntermediateOP
                {
                public:
                    static std::shared_ptr<IntermediateOP> translate(TRANS_ARGS)
                    {
                        std::shared_ptr<elementwise> inter_op(new elementwise());

                        if (out[0].get_size() == 0)
                        {
                            return inter_op;
                        }
                        else if (out.size() > 1)
                        {
                            throw std::runtime_error(
                                "Multi-output elementwise ops are not currently supported.");
                        }

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