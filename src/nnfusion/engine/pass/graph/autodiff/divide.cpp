#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Divide).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1) << "divide have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        // z = x / y, x_grad = z_grad / y, y_grad = -z_grad * x / (y * y) = -(z_grad / y) * (x / y) = -x_grad * z
        auto x = get_node_input(forward_node, 0);
        auto y = get_node_input(forward_node, 1);
        auto z = get_node_output(forward_node, 0);

        auto x_grad_op = std::make_shared<op::Divide>();
        x_grad_op->set_name(forward_node->get_name() + "_x_grad");
        auto x_grad = graph->add_node_and_edge(x_grad_op, {outputs_grad[0], y});
        auto neg_y_grad_op = std::make_shared<op::Multiply>();
        neg_y_grad_op->set_name(forward_node->get_name() + "_neg_y_grad");
        auto neg_y_grad = graph->add_node_and_edge(neg_y_grad_op, {GNodeIndex{x_grad}, z});
        auto y_grad_op = std::make_shared<op::Negative>();
        y_grad_op->set_name(forward_node->get_name() + "_y_grad");
        auto y_grad = graph->add_node_and_edge(y_grad_op, {GNodeIndex{neg_y_grad}});
        return GNodeIndexVector{GNodeIndex{x_grad}, GNodeIndex{y_grad}};
    });