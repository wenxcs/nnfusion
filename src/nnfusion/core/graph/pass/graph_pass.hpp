// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/graph/graph.hpp"
#include "graph_pass.hpp"

namespace nnfusion
{
    namespace graph
    {
        namespace pass
        {
            class GraphPass
            {
            public:
                GraphPass();
                ~GraphPass();

                bool run(std::shared_ptr<Graph> graph);
              
            };
        } //namespace pass
    } // namespace graph 
} // namespace nnfusion