// Microsoft (c) 2020, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class CSEPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
            };
        }
    }
}
