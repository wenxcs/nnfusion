// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "manager.hpp"

using namespace nnfusion::graph::pass;
using namespace std;

GraphPassManager::GraphPassManager()
{
}

GraphPassManager::~GraphPassManager()
{
}

void GraphPassManager::initialize_default_passes()
{
}

bool GraphPassManager::run_passes(std::shared_ptr<Graph> graph)
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

template <typename T, class... Args>
void GraphPassManager::register_pass(Args&&... args)
{
    static_assert(std::is_base_of<GraphPassBase, T>::value,
                  "pass not derived from graph pass base");
    auto pass = std::make_shared<T>(std::forward<Args>(args)...);
    auto pass_base = std::static_pointer_cast<GraphPassBase>(pass);
    m_pass_list.push_back(pass_base);
    m_pass_names.push_back(typeid(T).name());
}
