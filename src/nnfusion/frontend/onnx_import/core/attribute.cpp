//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "attribute.hpp"
#include "graph.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            std::vector<Graph> Attribute::get_graph_array() const
            {
                std::vector<Graph> result;
                for (const auto& graph : m_attribute_proto->graphs())
                {
                    result.emplace_back(graph);
                }
                return result;
            }

            Graph Attribute::get_graph() const { return Graph{m_attribute_proto->g()}; }
        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
