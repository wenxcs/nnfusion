// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"

namespace nnfusion
{
    namespace graph
    {
        namespace pass
        {
            class RuntimeConstantFoldingPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override { return true; }
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
