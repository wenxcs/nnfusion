#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Parameter).translator([](std::shared_ptr<GNode> forward_node,
                                                      const GNodeIndexVector& outputs_grad,
                                                      std::shared_ptr<nnfusion::graph::Graph> graph)
                                                       -> GNodeIndexVector {
    NNFUSION_CHECK(outputs_grad.size() == 1) << "parameter have only 1 output, but "
                                             << outputs_grad.size() << " outputs_grad provided";
    auto graph_outputs = graph->get_outputs();
    auto parameter_op = std::dynamic_pointer_cast<op::Parameter>(forward_node->get_op_ptr());
    ///\todo support other optimizer, support scheduled learning rate
    if (parameter_op->require_grad())
    {
        nnfusion::op::OpConfig::any myConfig;
        myConfig["learning_rate"] = 0.001;
        auto opt_op = std::make_shared<nnfusion::op::GenericOp>(
            forward_node->get_name() + "_sgd", "ApplyGradient", myConfig);
        auto opt_node =
            graph->add_node_and_edge(opt_op, {get_node_output(forward_node, 0), outputs_grad[0]});
        graph_outputs.push_back(opt_node);
    }
    else
    {
        graph_outputs.push_back(outputs_grad[0].gnode);
    }

    graph->set_outputs(graph_outputs);
    return GNodeIndexVector{};
});