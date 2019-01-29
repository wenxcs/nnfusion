// Microsoft (c) 2019, Wenxiang
#include "broadcast.hpp"

using namespace ngraph::runtime::gpu;
using namespace nnfusion::ir;

Broadcast::Broadcast(shared_ptr<Node> node)
    : Operator(node)
    , isMemcpy(false)
{
    auto broadcast = static_pointer_cast<ngraph::op::Broadcast>(node);
    assert_nullptr(broadcast) << "Node type is not Broadcast.";
    auto& axes = broadcast->get_broadcast_axes();
    if (axes.empty())
    {
        isMemcpy = true;
    }
    else
    {
        this->axes = AxisSet(axes);
    }

    arg_shape = args[0].get_shape();
    result_shape = out[0].get_shape();

    // calculate strides
    strides = ngraph::row_major_strides(result_shape);
    // precacluate invariants for integer division via multiplication
    stride_magic;
    stride_shift;
    for (int i = 0; i < strides.size(); i++)
    {
        int magic;
        int shift;
        std::tie(magic, shift) = idiv_magic_u64(strides[i]);
        stride_magic.push_back(magic);
        stride_shift.push_back(shift);
    }
    // calculate reduced tensor strides with 0s inserted for reduced axes
    reduced_shape = result_shape;
    for (auto const& axis : axes)
    {
        reduced_shape[axis] = 1;
    }
    reduced_strides = ngraph::row_major_strides(reduced_shape);
    for (auto const& axis : axes)
    {
        reduced_strides[axis] = 0;
    }

    rank = result_shape.size();
}

Operator_p Broadcast::translate(shared_ptr<Node> node)
{
    auto broadcast = static_pointer_cast<ngraph::op::Broadcast>(node);
    assert_nullptr(broadcast) << "Node type is not Broadcast.";
    auto& axes = broadcast->get_broadcast_axes();
    // If axes is empty, issue result op to do a memcopy
    if (axes.empty())
    {
        create_ptr(Result, inter_op, node);
        return inter_op;
    }
    else
    {
        create_ptr(Broadcast, inter_op, node);
        return inter_op;
    }
}