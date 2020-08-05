#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Log).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1) << "log have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        // y = log(x), x_grad = y_grad/x
        auto x = get_node_input(forward_node, 0);
        auto x_grad_op = std::make_shared<op::Divide>();
        x_grad_op->set_name(forward_node->get_name() + "_x_grad");
        auto x_grad = graph->add_node_and_edge(x_grad_op, {outputs_grad[0], x});
        return GNodeIndexVector{GNodeIndex{x_grad, 0}};
    });