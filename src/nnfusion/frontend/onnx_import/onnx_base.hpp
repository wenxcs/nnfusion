//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "onnx/onnx-ml.pb.h"

#include "nnfusion/common/common.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/frontend/frontend_base.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            using NamedNode = std::pair<std::string, std::shared_ptr<nnfusion::graph::GNode>>;
            using NamedNodeVector = std::vector<NamedNode>;
            using NodeMap =
                std::map<std::string, std::vector<std::shared_ptr<nnfusion::graph::GNode>>>;
            using ConvertFunc =
                std::function<NamedNodeVector(const onnx::NodeProto&,
                                              const NodeMap&,
                                              std::shared_ptr<nnfusion::graph::Graph> graph)>;

            using ConvertFuncMap =
                std::unordered_map<std::string, std::reference_wrapper<const ConvertFunc>>;

            inline void CopyToArray(const std::string& src, char* dst)
            {
                memcpy(dst, src.data(), src.size());
            }
        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
