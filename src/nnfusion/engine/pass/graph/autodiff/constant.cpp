#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Constant).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1) << "constant have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        // do nothing
        return GNodeIndexVector{};
    });