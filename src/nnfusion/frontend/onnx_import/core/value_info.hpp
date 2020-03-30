//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "../util/util.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            class ValueInfo
            {
            public:
                ValueInfo(ValueInfo&&) = default;
                ValueInfo(const ValueInfo&) = default;

                ValueInfo() = delete;
                explicit ValueInfo(const onnx::ValueInfoProto& value_info_proto)
                    : m_value_info_proto{&value_info_proto}
                {
                    NNFUSION_CHECK(value_info_proto.type().has_tensor_type())
                        << "value info has no tensor type specified.";

                    for (const auto& dim : value_info_proto.type().tensor_type().shape().dim())
                    {
                        m_shape.emplace_back(static_cast<Shape::value_type>(dim.dim_value()));
                    }
                    NNFUSION_CHECK(m_value_info_proto->type().tensor_type().has_elem_type())
                        << "value info has no element type specified.";
                    ONNXDataTypeToNNFusionElementType(
                        m_value_info_proto->type().tensor_type().elem_type(), &m_type);
                }

                ValueInfo& operator=(const ValueInfo&) = delete;
                ValueInfo& operator=(ValueInfo&&) = delete;

                const std::string& get_name() const { return m_value_info_proto->name(); }
                const Shape& get_shape() const { return m_shape; }
                const element::Type& get_element_type() const { return m_type; }
            private:
                const onnx::ValueInfoProto* m_value_info_proto;
                Shape m_shape;
                element::Type m_type;
            };

            inline std::ostream& operator<<(std::ostream& outs, const ValueInfo& info)
            {
                return (outs << "<ValueInfo: " << info.get_name() << ">");
            }
        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
