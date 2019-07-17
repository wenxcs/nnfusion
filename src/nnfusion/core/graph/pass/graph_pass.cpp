// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass.hpp"
#include "constant_folding_pass.hpp"
#include "manager.hpp"
#include "reshape_inplace_pass.hpp"

using namespace nnfusion::graph::pass;
using namespace std;

bool GraphPass::run(std::shared_ptr<Graph> graph)
{
    GraphPassManager pass_manager;
    pass_manager.register_pass<ConstantFoldingPass>();
    pass_manager.register_pass<ReshapeInplacePass>();
    pass_manager.run_passes(graph);

    return true;
}