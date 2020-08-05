#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Tanh).translator([](std::shared_ptr<GNode> forward_node,
                                                 const GNodeIndexVector& outputs_grad,
                                                 std::shared_ptr<nnfusion::graph::Graph> graph)
                                                  -> GNodeIndexVector {
    NNFUSION_CHECK(outputs_grad.size() == 1) << "tanh have only 1 output, but "
                                             << outputs_grad.size() << " outputs_grad provided";
    // f = tanh(x), x_grad = f_grad - f_grad * f**2
    auto f = get_node_output(forward_node, 0);
    auto f_square_op = std::make_shared<op::Square>();
    f_square_op->set_name(forward_node->get_name() + "_f_square");
    auto f_square = graph->add_node_and_edge(f_square_op, {f});
    auto f_grad_mul_f_square_op = std::make_shared<op::Multiply>();
    f_grad_mul_f_square_op->set_name(forward_node->get_name() + "_f_grad_mul_f_square");
    auto f_grad_mul_f_square =
        graph->add_node_and_edge(f_grad_mul_f_square_op, {outputs_grad[0], GNodeIndex{f_square}});
    auto x_grad_op = std::make_shared<op::Subtract>();
    x_grad_op->set_name(forward_node->get_name() + "_x_grad");
    auto x_grad =
        graph->add_node_and_edge(x_grad_op, {outputs_grad[0], GNodeIndex{f_grad_mul_f_square}});

    return GNodeIndexVector{GNodeIndex{x_grad, 0}};
});