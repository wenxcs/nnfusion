//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------
#pragma once

#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/frontend/frontend_base.hpp"

#include "torch/script.h"

namespace nnfusion
{
    namespace frontend
    {
        namespace torchscript_import
        {
            using GNodePtr = std::shared_ptr<nnfusion::graph::GNode>;
            using TNodePtr = torch::jit::Node*;
            using TBlockPtr = torch::jit::Block*;
            using TValuePtr = torch::jit::Value*;
            using NodeMap = std::unordered_map<TNodePtr, graph::GNodeVector>;
            using ConvertFunc =
                std::function<graph::GNodeVector(const TNodePtr n,
                                                 NodeMap& tnode2gnodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)>;

        } // namespace torchscript_import
    }     // namespace frontend
} // namespace nnfusion
