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
            class GraphPassManager
            {
            public:
                GraphPassManager();
                ~GraphPassManager();

                void initialize_default_passes();

                template <typename T, class... Args>
                void register_pass(Args&&... args)
                {
                    static_assert(std::is_base_of<GraphPass, T>::value, "pass not derived from graph pass base");
                    auto pass = std::make_shared<T>(std::forward<Args>(args)...);
                    auto pass_base = std::static_pointer_cast<GraphPass>(pass);
                    m_pass_list.push_back(pass_base);
                    m_pass_names.push_back(typeid(T).name());

                }
                
                bool run_passes(std::shared_ptr<Graph> graph)
                {
                    bool status = true;
                    for (auto& pass : m_pass_list)
                    {
                        status = pass->run_on_graph(graph);
                        if (!status)
                            break;
                    }
                    return status;
                }

            private:
                std::vector<std::string> m_pass_names;
                std::vector<std::shared_ptr<GraphPass>> m_pass_list;                 
            };
        } //namespace pass
    } // namespace graph 
} // namespace nnfusion