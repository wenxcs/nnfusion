#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Softmax).translator([](std::shared_ptr<GNode> forward_node,
                                                    const GNodeIndexVector& outputs_grad,
                                                    std::shared_ptr<nnfusion::graph::Graph> graph)
                                                     -> GNodeIndexVector {
    NNFUSION_CHECK(outputs_grad.size() == 1) << "softmax have only 1 output, but "
                                             << outputs_grad.size() << " outputs_grad provided";
    auto softmax_op = std::dynamic_pointer_cast<op::Softmax>(forward_node->get_op_ptr());
    auto axis = softmax_op->get_axes();
    bool in_log_space = softmax_op->is_in_log_space();
    auto x_grad_op = std::make_shared<op::SoftmaxGrad>(axis, in_log_space);
    x_grad_op->set_name(forward_node->get_name() + "_x_grad");
    auto x_grad =
        graph->add_node_and_edge(x_grad_op, {outputs_grad[0], get_node_output(forward_node, 0)});
    return GNodeIndexVector{GNodeIndex{x_grad, 0}};
});