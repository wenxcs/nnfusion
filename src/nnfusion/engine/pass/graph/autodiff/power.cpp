#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Power).translator([](std::shared_ptr<GNode> forward_node,
                                                  const GNodeIndexVector& outputs_grad,
                                                  std::shared_ptr<nnfusion::graph::Graph> graph)
                                                   -> GNodeIndexVector {
    NNFUSION_CHECK(outputs_grad.size() == 1) << "power have only 1 output, but "
                                             << outputs_grad.size() << " outputs_grad provided";
    // z = x**y, x_grad = z_grad * y * x ** (y - 1) = z_grad * y * z / x, y_grad = z_grad * x ** y * lnx = z_grad * z * lnx
    auto x = get_node_input(forward_node, 0);
    auto y = get_node_input(forward_node, 1);
    auto z = get_node_output(forward_node, 0);
    auto z_grad_mul_z =
        graph->add_node_and_edge(std::make_shared<op::Multiply>(), {outputs_grad[0], z});
    auto y_div_x = graph->add_node_and_edge(std::make_shared<op::Divide>(), {y, x});
    auto lnx = graph->add_node_and_edge(std::make_shared<op::Log>(), {x});
    auto x_grad =
        graph->add_node_and_edge(std::make_shared<op::Multiply>(), {z_grad_mul_z, y_div_x});
    auto y_grad = graph->add_node_and_edge(std::make_shared<op::Multiply>(), {z_grad_mul_z, lnx});
    return GNodeIndexVector{GNodeIndex{x_grad}, GNodeIndex{y_grad}};
});