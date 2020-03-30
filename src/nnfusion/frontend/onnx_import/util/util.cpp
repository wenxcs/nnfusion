//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "util.hpp"
#include "../core/tensor.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            bool ONNXDataTypeToNNFusionElementType(const onnx::TensorProto_DataType onnx_dt,
                                                   nnfusion::element::Type* nnfusion_et)
            {
                switch (onnx_dt)
                {
                case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
                    *nnfusion_et = element::boolean;
                    break;
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16:
                    *nnfusion_et = element::f32;
                    break;
                case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
                    *nnfusion_et = element::f64;
                    break;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT8:
                    *nnfusion_et = element::i8;
                    break;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT16:
                    *nnfusion_et = element::i16;
                    break;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
                    *nnfusion_et = element::i32;
                    break;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
                    *nnfusion_et = element::i64;
                    break;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT8:
                    *nnfusion_et = element::u8;
                    break;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT16:
                    *nnfusion_et = element::u16;
                    break;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT32:
                    *nnfusion_et = element::u32;
                    break;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT64:
                    *nnfusion_et = element::u64;
                    break;
                default:
                    NNFUSION_CHECK_FAIL() << "unsupported onnx element type: "
                                          << onnx::TensorProto_DataType_Name(onnx_dt);
                    return false;
                }
                return true;
            }

            template <typename T>
            std::shared_ptr<op::Constant>
                make_constant_op(const element::Type& type, const Shape shape, const Tensor& tensor)
            {
                return std::make_shared<op::Constant>(type, shape, tensor.get_data<T>());
            }

            std::shared_ptr<op::Constant> make_constant_op(const onnx::TensorProto_DataType onnx_et,
                                                           const Shape shape,
                                                           const Tensor& tensor)
            {
                switch (onnx_et)
                {
                case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
                    return make_constant_op<bool>(element::boolean, shape, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16:
                    return make_constant_op<float>(element::f32, shape, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
                    return make_constant_op<double>(element::f64, shape, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_INT8:
                    return make_constant_op<int8_t>(element::i8, shape, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_INT16:
                    return make_constant_op<int16_t>(element::i16, shape, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
                    return make_constant_op<int32_t>(element::i32, shape, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
                    return make_constant_op<int64_t>(element::i64, shape, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT8:
                    return make_constant_op<uint8_t>(element::u8, shape, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT16:
                    return make_constant_op<uint16_t>(element::u16, shape, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT32:
                    return make_constant_op<uint32_t>(element::u32, shape, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT64:
                    return make_constant_op<uint64_t>(element::u64, shape, tensor);
                default:
                    NNFUSION_CHECK_FAIL() << "unsupported value info element type: "
                                          << onnx::TensorProto_DataType_Name(onnx_et);
                }
            }

            std::shared_ptr<graph::GNode> GetInputNode(const NodeMap& all_ng_nodes,
                                                       const onnx::NodeProto& node,
                                                       size_t input_idx)
            {
                std::shared_ptr<graph::GNode> result = nullptr;
                try
                {
                    result = all_ng_nodes.at(node.input(input_idx)).at(0);
                }
                catch (const std::out_of_range&)
                {
                    NNFUSION_CHECK_FAIL() << "Input Ngraph op not found for "
                                          << node.input(input_idx);
                }
                return result;
            }

            graph::GNodeVector GetAllInputNode(const NodeMap& all_ng_nodes,
                                               const onnx::NodeProto& node)
            {
                graph::GNodeVector nodes;
                for (size_t i = 0; i < node.input_size(); i++)
                {
                    nodes.push_back(GetInputNode(all_ng_nodes, node, i));
                }
                return nodes;
            }

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
