#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(OneHot).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        // OneHot no backprop
        return GNodeIndexVector{GNodeIndex{}};
    });