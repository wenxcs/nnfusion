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
            class Tensor
            {
            public:
                enum class Type
                {
                    undefined = onnx::TensorProto_DataType_UNDEFINED,
                    float32 = onnx::TensorProto_DataType_FLOAT,
                    uint8 = onnx::TensorProto_DataType_UINT8,
                    int8 = onnx::TensorProto_DataType_INT8,
                    uint16 = onnx::TensorProto_DataType_UINT16,
                    int16 = onnx::TensorProto_DataType_INT16,
                    int32 = onnx::TensorProto_DataType_INT32,
                    int64 = onnx::TensorProto_DataType_INT64,
                    string = onnx::TensorProto_DataType_STRING,
                    boolean = onnx::TensorProto_DataType_BOOL,
                    float16 = onnx::TensorProto_DataType_FLOAT16,
                    float64 = onnx::TensorProto_DataType_DOUBLE,
                    uint32 = onnx::TensorProto_DataType_UINT32,
                    uint64 = onnx::TensorProto_DataType_UINT64,
                    complex64 = onnx::TensorProto_DataType_COMPLEX64,
                    complex128 = onnx::TensorProto_DataType_COMPLEX128
                };

                Tensor() = delete;
                explicit Tensor(const onnx::TensorProto& tensor)
                    : m_tensor_proto{&tensor}
                    , m_shape{std::begin(tensor.dims()), std::end(tensor.dims())}
                {
                }

                Tensor(const Tensor&) = default;
                Tensor(Tensor&&) = default;

                Tensor& operator=(const Tensor&) = delete;
                Tensor& operator=(Tensor&&) = delete;

                const Shape& get_shape() const { return m_shape; }
                template <typename T>
                std::vector<T> get_data() const
                {
                    NNFUSION_CHECK(!m_tensor_proto->has_segment())
                        << "loading tensor segments not supported.";

                    return detail::get_data<T>(*m_tensor_proto);
                }

                const std::string& get_name() const
                {
                    NNFUSION_CHECK(m_tensor_proto->has_name()) << "tensor has no name specified.";
                    return m_tensor_proto->name();
                }

                Type get_type() const
                {
                    NNFUSION_CHECK(m_tensor_proto->has_data_type())
                        << "tensor has no data type specified.";
                    return static_cast<Type>(m_tensor_proto->data_type());
                }

                const element::Type& get_ng_type() const
                {
                    NNFUSION_CHECK(m_tensor_proto->has_data_type())
                        << "tensor has no data type specified.";

                    switch (m_tensor_proto->data_type())
                    {
                    case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
                        return element::boolean;
                    case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
                    case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16:
                        return element::f32;
                    case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
                        return element::f64;
                    case onnx::TensorProto_DataType::TensorProto_DataType_INT8: return element::i8;
                    case onnx::TensorProto_DataType::TensorProto_DataType_INT16:
                        return element::i16;
                    case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
                        return element::i32;
                    case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
                        return element::i64;
                    case onnx::TensorProto_DataType::TensorProto_DataType_UINT8: return element::u8;
                    case onnx::TensorProto_DataType::TensorProto_DataType_UINT16:
                        return element::u16;
                    case onnx::TensorProto_DataType::TensorProto_DataType_UINT32:
                        return element::u32;
                    case onnx::TensorProto_DataType::TensorProto_DataType_UINT64:
                        return element::u64;
                    case onnx::TensorProto_DataType::TensorProto_DataType_UNDEFINED:
                        NNFUSION_CHECK_FAIL() << "data type is not defined";
                        break;
                    default:
                        NNFUSION_CHECK_FAIL()
                            << "unsupported data type: "
                            << onnx::TensorProto_DataType_Name(m_tensor_proto->data_type());
                        break;
                    }
                }

                operator onnx::TensorProto_DataType() const { return m_tensor_proto->data_type(); }
            private:
                const onnx::TensorProto* m_tensor_proto;
                Shape m_shape;
            };

            inline std::ostream& operator<<(std::ostream& outs, const Tensor& tensor)
            {
                return (outs << "<Tensor: " << tensor.get_name() << ">");
            }

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfuison
