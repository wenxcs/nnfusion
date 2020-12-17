#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Broadcast).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1) << "broadcast have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        auto broadcast_op = std::dynamic_pointer_cast<op::Broadcast>(forward_node->get_op_ptr());
        auto sum_op = std::make_shared<op::Sum>(broadcast_op->get_broadcast_axes());
        auto x_grad = graph->add_node_and_edge(sum_op, {outputs_grad[0]});
        return GNodeIndexVector{GNodeIndex{x_grad, 0}};
    });