#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Parameter).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1) << "parameter have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        ///\todo add optimizer
        auto graph_outputs = graph->get_outputs();
        graph_outputs.push_back(outputs_grad[0].gnode);
        graph->set_outputs(graph_outputs);
        return GNodeIndexVector{};
    });