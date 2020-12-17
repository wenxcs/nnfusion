#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Add).translator(
    [](std::shared_ptr<graph::GNode> forward_node,
       const graph::GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> graph::GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1) << "add have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        return graph::GNodeIndexVector{outputs_grad[0], outputs_grad[0]};
    });
