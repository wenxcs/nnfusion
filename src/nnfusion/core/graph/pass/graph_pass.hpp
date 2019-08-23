// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace graph
    {
        namespace pass
        {
            class GraphPass
            {
            public:
                bool run(std::shared_ptr<Graph> graph);
            };
        } //namespace pass
    }     // namespace graph
} // namespace nnfusion