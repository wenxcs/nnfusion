//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "../onnx_base.hpp"
#include "nnfusion/common/common.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace detail
            {
                template <typename T, typename Container>
                inline std::vector<T> __get_data(const Container& container)
                {
                    return {std::begin(container), std::end(container)};
                }

                template <typename T>
                inline std::vector<T> __get_raw_data(const std::string& raw_data)
                {
                    auto it = reinterpret_cast<const T*>(raw_data.data());
                    return {it, it + (raw_data.size() / sizeof(T))};
                }

                template <typename T>
                inline std::vector<T> get_data(const onnx::TensorProto& tensor)
                {
                    NNFUSION_CHECK_FAIL() << "unsupported data type: "
                                          << onnx::TensorProto_DataType_Name(tensor.data_type());
                }

                template <>
                inline std::vector<double> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return __get_raw_data<double>(tensor.raw_data());
                    }
                    switch (tensor.data_type())
                    {
                    case onnx::TensorProto_DataType_DOUBLE:
                        return __get_data<double>(tensor.double_data());
                    case onnx::TensorProto_DataType_FLOAT:
                    case onnx::TensorProto_DataType_FLOAT16:
                        return __get_data<double>(tensor.float_data());
                    case onnx::TensorProto_DataType_INT32:
                        return __get_data<double>(tensor.int32_data());
                    case onnx::TensorProto_DataType_INT64:
                        return __get_data<double>(tensor.int64_data());
                    case onnx::TensorProto_DataType_UINT64:
                        return __get_data<double>(tensor.uint64_data());
                    default:
                        NNFUSION_CHECK_FAIL()
                            << "invalid data type: "
                            << onnx::TensorProto_DataType_Name(tensor.data_type());
                        break;
                    }
                }

                template <>
                inline std::vector<float> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return __get_raw_data<float>(tensor.raw_data());
                    }
                    if ((tensor.data_type() == onnx::TensorProto_DataType_FLOAT) ||
                        (tensor.data_type() == onnx::TensorProto_DataType_FLOAT16))
                    {
                        return __get_data<float>(tensor.float_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_INT32)
                    {
                        return __get_data<float>(tensor.int32_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_INT64)
                    {
                        return __get_data<float>(tensor.int64_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_UINT64)
                    {
                        return __get_data<float>(tensor.uint64_data());
                    }
                    NNFUSION_CHECK_FAIL() << "invalid data type: "
                                          << onnx::TensorProto_DataType_Name(tensor.data_type());
                }

                template <>
                inline std::vector<int32_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return __get_raw_data<int32_t>(tensor.raw_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_INT32)
                    {
                        return __get_data<int32_t>(tensor.int32_data());
                    }
                    NNFUSION_CHECK_FAIL() << "invalid data type: "
                                          << onnx::TensorProto_DataType_Name(tensor.data_type());
                }

                template <>
                inline std::vector<int64_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return __get_raw_data<int64_t>(tensor.raw_data());
                    }
                    NNFUSION_CHECK(tensor.data_type() == onnx::TensorProto_DataType_INT64);

                    return __get_data<int64_t>(tensor.int64_data());
                }

                template <>
                inline std::vector<uint64_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return __get_raw_data<uint64_t>(tensor.raw_data());
                    }
                    NNFUSION_CHECK(tensor.data_type() == onnx::TensorProto_DataType_UINT64)
                        << "invalid data type: "
                        << onnx::TensorProto_DataType_Name(tensor.data_type());
                    return __get_data<uint64_t>(tensor.uint64_data());
                }
            }

            class Tensor;

            bool ONNXDataTypeToNNFusionElementType(const onnx::TensorProto_DataType onnx_dt,
                                                   nnfusion::element::Type* nnfusion_et);

            template <typename T>
            std::shared_ptr<op::Constant> make_constant_op(const element::Type& type,
                                                           const Shape shape,
                                                           const Tensor& tensor);

            std::shared_ptr<op::Constant> make_constant_op(const onnx::TensorProto_DataType onnx_et,
                                                           const Shape shape,
                                                           const Tensor& tensor);

            std::shared_ptr<graph::GNode> GetInputNode(const NodeMap& all_ng_nodes,
                                                       const onnx::NodeProto& node,
                                                       size_t input_idx);

            graph::GNodeVector GetAllInputNode(const NodeMap& all_ng_nodes,
                                               const onnx::NodeProto& node);
        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
