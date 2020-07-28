#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Multiply).translator([](std::shared_ptr<GNode> forward_node,
                                                     const GNodeIndexVector& outputs_grad,
                                                     std::shared_ptr<nnfusion::graph::Graph> graph)
                                                      -> GNodeIndexVector {
    NNFUSION_CHECK(outputs_grad.size() == 1) << "multiply have only 1 output, but "
                                             << outputs_grad.size() << " outputs_grad provided";
    auto x = get_node_input(forward_node, 0);
    auto y = get_node_input(forward_node, 1);
    auto x_grad = graph->add_node_and_edge(std::make_shared<op::Multiply>(), {y, outputs_grad[0]});
    auto y_grad = graph->add_node_and_edge(std::make_shared<op::Multiply>(), {x, outputs_grad[0]});
    return GNodeIndexVector{GNodeIndex{x_grad}, GNodeIndex{y_grad}};
});