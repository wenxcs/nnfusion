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
        // Convert an ONNX model to a nnfusion graph (input stream)
        std::shared_ptr<nnfusion::graph::Graph> load_onnx_model(std::istream&);

        // Convert an ONNX model to a nnfusion graph
        std::shared_ptr<nnfusion::graph::Graph> load_onnx_model(const std::string&);
    } // namespace frontend
} // namespace nnfusion
