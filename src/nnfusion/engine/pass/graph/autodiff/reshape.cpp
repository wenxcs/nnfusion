#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Reshape).translator([](std::shared_ptr<GNode> forward_node,
                                                    const GNodeIndexVector& outputs_grad,
                                                    std::shared_ptr<nnfusion::graph::Graph> graph)
                                                     -> GNodeIndexVector {
    NNFUSION_CHECK(outputs_grad.size() == 1) << "reshape have only 1 output, but "
                                             << outputs_grad.size() << " outputs_grad provided";
    auto reshape_op = std::dynamic_pointer_cast<op::Reshape>(forward_node->get_op_ptr());
    ///\todo handle transpose case
    NNFUSION_CHECK(!reshape_op->get_is_transpose());
    auto backprop_op = std::make_shared<op::Reshape>(get_default_order(outputs_grad[0].get_shape()),
                                                     forward_node->get_input_shape(0));
    auto x_grad = graph->add_node_and_edge(backprop_op, {outputs_grad[0]});
    return GNodeIndexVector{GNodeIndex{x_grad, 0}};
});