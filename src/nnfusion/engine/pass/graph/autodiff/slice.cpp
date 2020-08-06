#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Slice).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        ///\todo support slice backward
        NNFUSION_LOG(NNFUSION_WARNING) << "Slice backward not implemented yet.";
        return GNodeIndexVector{pass::graph::autodiff::DiffEngine::EMPTY_GNODE_INDEX};
    });