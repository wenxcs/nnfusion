//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace frontend
    {
        struct ParamInfo
        {
            nnfusion::Shape shape;
            nnfusion::element::Type type;
            ParamInfo(const nnfusion::Shape&, nnfusion::element::Type);
            ParamInfo(const nnfusion::Shape&, const std::string&);
            ParamInfo(const std::string&);
        };

        std::vector<ParamInfo> build_params_from_string(const std::string&);

    } // namespace frontend
} // namespace nnfusion