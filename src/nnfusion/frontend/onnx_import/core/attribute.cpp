//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "attribute.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            std::vector<onnx::GraphProto> Attribute::get_graphproto_array() const
            {
                std::vector<onnx::GraphProto> result;
                for (const auto& graphproto : m_attribute_proto->graphs())
                {
                    result.emplace_back(graphproto);
                }
                return result;
            }

            onnx::GraphProto Attribute::get_graphproto() const { return m_attribute_proto->g(); }
        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
