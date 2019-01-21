// Microsoft (c) 2019, Wenxiang
#include "ngraph/runtime/nnfusion/intermediate/Result.hpp"

using namespace ngraph;
using namespace ngraph::runtime::nnfusion::intermediate;

Result::Result(shared_ptr<Node> node)
    : IntermediateOP(node)
{
}

std::shared_ptr<IntermediateOP> Result::translate(shared_ptr<Node> node)
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