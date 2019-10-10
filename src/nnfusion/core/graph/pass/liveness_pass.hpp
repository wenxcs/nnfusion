// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
namespace nnfusion
{
    namespace graph
    {
        namespace pass
        {
            class LivenessPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override;
                const std::unordered_set<std::string> memory_reuse_blacklist =
                    {}; //{"StridedSliceGrad"};
            };
        }
    }
}