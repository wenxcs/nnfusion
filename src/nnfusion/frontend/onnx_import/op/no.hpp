//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "core/node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateNoOp(const onnx::NodeProto& node_proto,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph);
            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace nnfusion