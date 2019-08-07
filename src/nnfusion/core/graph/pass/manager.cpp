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