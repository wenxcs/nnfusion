#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(AddN).translator(
    [](std::shared_ptr<graph::GNode> forward_node,
       const graph::GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> graph::GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1) << "addn have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        return graph::GNodeIndexVector(forward_node->get_input_size(), outputs_grad[0]);
    });
