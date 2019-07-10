// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace graph
    {
        namespace pass
        {
            class GraphPassManager
            {
            public:
                GraphPassManager();
                ~GraphPassManager();

                void initialize_default_passes();

                template <typename T, class... Args>
                void register_pass(Args&&... args);

                bool run_passes(std::shared_ptr<Graph> graph);

            private:
                std::vector<std::string> m_pass_names;
                std::vector<std::shared_ptr<GraphPassBase>> m_pass_list;
            };
        } //namespace pass
    }     // namespace graph
} // namespace nnfusion