//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <string>

#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace frontend
    {
        // Convert an ONNX model to a nnfusion graph
        std::shared_ptr<nnfusion::graph::Graph>
            load_onnx_model(const std::string&,
                            const std::unordered_map<std::string, size_t>& dim_params = {});
    } // namespace frontend
} // namespace nnfusion
