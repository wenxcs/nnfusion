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
                NamedNodeVector TranslateGatherOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph);

                NamedNodeVector
                    TranslateGatherNDOp(const onnx::NodeProto& node_proto,
                                        const NodeMap& all_ng_nodes,
                                        std::shared_ptr<nnfusion::graph::Graph> m_graph);

                NamedNodeVector
                    TranslateGatherGradOp(const onnx::NodeProto& node_proto,
                                          const NodeMap& all_ng_nodes,
                                          std::shared_ptr<nnfusion::graph::Graph> m_graph);

                NamedNodeVector
                    TranslateGatherNDGradOp(const onnx::NodeProto& node_proto,
                                            const NodeMap& all_ng_nodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph);
            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
