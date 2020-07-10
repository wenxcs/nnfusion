//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "core/node.hpp"
#include "no.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateIdentityOp(const onnx::NodeProto& node_proto,
                                                    const NodeMap& all_ng_nodes,
                                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    NNFUSION_CHECK(node_proto.input_size() == 1 && node_proto.output_size() == 1);
                    return TranslateNoOp(node_proto, all_ng_nodes, m_graph);
                }
            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
