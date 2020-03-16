//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <string>

#include "../util/parameter.hpp"
#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace frontend
    {
        // Convert a TorchScript model to a nnfusion graph
        std::shared_ptr<nnfusion::graph::Graph> load_torchscript_model(const std::string&);

        std::shared_ptr<nnfusion::graph::Graph>
            load_torchscript_model(const std::string&, const std::vector<ParamInfo>&);
    } // namespace frontend

} // namespace nnfusion
