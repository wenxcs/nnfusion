// Microsoft (c) 2020, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_bool(fautodiff);

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class AutodiffPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
            };
        }
    }
}
