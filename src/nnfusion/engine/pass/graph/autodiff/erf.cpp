#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Erf).translator([](std::shared_ptr<GNode> forward_node,
                                                const GNodeIndexVector& outputs_grad,
                                                std::shared_ptr<nnfusion::graph::Graph> graph)
                                                 -> GNodeIndexVector {
    NNFUSION_CHECK(outputs_grad.size() == 1) << "erf have only 1 output, but "
                                             << outputs_grad.size() << " outputs_grad provided";
    // y = erf(x), x_grad = y_grad * (2 / sqrt(pi)) * exp ** (-x**2)
    auto x = get_node_input(forward_node, 0);
    auto y_grad = outputs_grad.at(0);

    // x_grad
    const float two_sqrt_pi = 1.12837916709551257390; /* 2/sqrt(pi) */

    auto square_x = graph->add_node_and_edge(std::make_shared<op::Square>(), {x});
    auto neg_square_x = graph->add_node_and_edge(std::make_shared<op::Negative>(), {square_x});
    auto exp_neg_square_x = graph->add_node_and_edge(std::make_shared<op::Exp>(), {neg_square_x});
    auto two_sqrt_pi_op = std::make_shared<op::Constant>(
        element::f32, x.get_shape(), std::vector<float>{two_sqrt_pi});
    auto two_sqrt_pi_gnode =
        graph->add_node_and_edge(two_sqrt_pi_op, nnfusion::graph::GNodeVector({}));
    auto erf_grad = graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                             {two_sqrt_pi_gnode, exp_neg_square_x});
    auto x_grad =
        graph->add_node_and_edge(std::make_shared<op::Multiply>(), {y_grad, GNodeIndex{erf_grad}});

    return GNodeIndexVector{GNodeIndex{x_grad, 0}};
});