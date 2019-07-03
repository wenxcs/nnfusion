// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace graph
    {
        namespace pass
        {
            class GraphPassBase
            {
            public:
                virtual ~GraphPassBase() {}

                virtual bool run_on_graph(std::shared_ptr<Graph>& graph) = 0;
            };

        }
    }
}


