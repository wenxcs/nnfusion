//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "no.hpp"

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
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    NamedNodeVector ret;
                    size_t input_cnt = node_proto.input_size();
                    size_t output_cnt = node_proto.output_size();
                    NNFUSION_CHECK(input_cnt == output_cnt);
                    for (int i = 0; i < input_cnt; i++)
                    {
                        auto input_index = GetInputIndex(all_ng_nodes, node_proto, i);
                        ret.emplace_back(node_proto.output(i), input_index);
                    }
                    return ret;
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace nnfusion
