#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Subtract).translator([](std::shared_ptr<GNode> forward_node,
                                                     const GNodeIndexVector& outputs_grad,
                                                     std::shared_ptr<nnfusion::graph::Graph> graph)
                                                      -> GNodeIndexVector {
    NNFUSION_CHECK(outputs_grad.size() == 1) << "subtract have only 1 output, but "
                                             << outputs_grad.size() << " outputs_grad provided";
    auto neg_out_grad = graph->add_node_and_edge(std::make_shared<op::Negative>(), outputs_grad);
    return GNodeIndexVector{outputs_grad[0], GNodeIndex{neg_out_grad, 0}};
});