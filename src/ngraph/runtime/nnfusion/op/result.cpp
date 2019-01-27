// Microsoft (c) 2019, Wenxiang
#include "result.hpp"

using namespace nnfusion::ir;

Result::Result(shared_ptr<Node> node)
    : Operator(node)
{
}

Operator_p Result::translate(shared_ptr<Node> node)
{
    create_ptr(Result, inter_op, node);

    if (inter_op->args[0].get_name() == inter_op->out[0].get_name())
    {
        create_ptr(Noop, notrans, node);
        NGRAPH_DEBUG << "Skipping translation for " << node->get_name() << "\n";
        return notrans;
    }

    return inter_op;
}