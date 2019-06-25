//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// This macro is use to determine if a compiler in use:
//    1. In editor: Use the protobuf files in proto/ for code completion
//    2. In compiling: Use auto-generated probobuf file, Read proto/CmakeLists.txt
//       for details.
#ifdef __cplusplus
#include "graph.pb.h"
#else
#include "../proto/graph.pb.h"
#endif

#include "../tensorflow_base.hpp"
#include "ngraph/function.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
namespace ngraph
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            static const int kControlSlot = -1;
            struct TensorId : public std::pair<std::string, int>
            {
                TensorId() {}
                TensorId(const std::string& str, int idx)
                {
                    first = str;
                    second = idx;
                }
                TensorId(const TensorId& id)
                    : TensorId(id.first, id.second)
                {
                }

                const std::string& node() const { return first; }
                int index() const { return second; }
                std::string ToString() const
                {
                    if (second == kControlSlot)
                        return "^" + first;
                    return first + ":" + std::to_string(second);
                }
            };

            // Validates type T for whether it is a supported DataType.
            template <class T>
            struct IsValidDataType;

            // DataTypeToEnum<T>::v() and DataTypeToEnum<T>::value are the DataType
            // constants for T, e.g. DataTypeToEnum<float>::v() is DT_FLOAT.
            template <class T>
            struct DataTypeToEnum
            {
                static_assert(IsValidDataType<T>::value, "Specified Data Type not supported");
            }; // Specializations below

            // EnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
            // EnumToDataType<DT_FLOAT>::Type is float.
            template <tensorflow::DataType VALUE>
            struct EnumToDataType
            {
            }; // Specializations below

// Template specialization for both DataTypeToEnum and EnumToDataType.
#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)                                                            \
    template <>                                                                                    \
    struct DataTypeToEnum<TYPE>                                                                    \
    {                                                                                              \
        static tensorflow::DataType v() { return ENUM; }                                           \
        static constexpr tensorflow::DataType value = ENUM;                                        \
    };                                                                                             \
    template <>                                                                                    \
    struct IsValidDataType<TYPE>                                                                   \
    {                                                                                              \
        static constexpr bool value = true;                                                        \
    };                                                                                             \
    template <>                                                                                    \
    struct EnumToDataType<ENUM>                                                                    \
    {                                                                                              \
        typedef TYPE Type;                                                                         \
    }

            MATCH_TYPE_AND_ENUM(float, tensorflow::DataType::DT_FLOAT);
            MATCH_TYPE_AND_ENUM(double, tensorflow::DataType::DT_DOUBLE);
            MATCH_TYPE_AND_ENUM(int32, tensorflow::DataType::DT_INT32);
            MATCH_TYPE_AND_ENUM(uint32, tensorflow::DataType::DT_UINT32);
            MATCH_TYPE_AND_ENUM(uint16, tensorflow::DataType::DT_UINT16);
            MATCH_TYPE_AND_ENUM(uint8, tensorflow::DataType::DT_UINT8);
            MATCH_TYPE_AND_ENUM(int16, tensorflow::DataType::DT_INT16);
            MATCH_TYPE_AND_ENUM(int8, tensorflow::DataType::DT_INT8);
            MATCH_TYPE_AND_ENUM(std::string, tensorflow::DataType::DT_STRING);
            //MATCH_TYPE_AND_ENUM(complex64, tensorflow::DataType::DT_COMPLEX64);
            //MATCH_TYPE_AND_ENUM(complex128, tensorflow::DataType::DT_COMPLEX128);
            //MATCH_TYPE_AND_ENUM(qint8, tensorflow::DataType::DT_QINT8);
            //MATCH_TYPE_AND_ENUM(quint8, tensorflow::DataType::DT_QUINT8);
            //MATCH_TYPE_AND_ENUM(qint16, tensorflow::DataType::DT_QINT16);
            //MATCH_TYPE_AND_ENUM(quint16, tensorflow::DataType::DT_QUINT16);
            //MATCH_TYPE_AND_ENUM(qint32, tensorflow::DataType::DT_QINT32);
            //MATCH_TYPE_AND_ENUM(bfloat16, tensorflow::DataType::DT_BFLOAT16);
            //MATCH_TYPE_AND_ENUM(Eigen::half, tensorflow::DataType::DT_HALF);
            //MATCH_TYPE_AND_ENUM(ResourceHandle, tensorflow::DataType::DT_RESOURCE);
            //MATCH_TYPE_AND_ENUM(Variant, tensorflow::DataType::DT_VARIANT);
            MATCH_TYPE_AND_ENUM(int64, tensorflow::DataType::DT_INT64);
            MATCH_TYPE_AND_ENUM(uint64, tensorflow::DataType::DT_UINT64);
            MATCH_TYPE_AND_ENUM(bool, tensorflow::DataType::DT_BOOL);

#undef MATCH_TYPE_AND_ENUM

            // Template specialization for both DataTypeToEnum and EnumToDataType.
            // Converts a TensorFlow DataType to an nGraph element::Type. Returns
            // false if the element type is not supported by nGraph
            // Core. Otherwise returns true.
            bool TFDataTypeToNGraphElementType(const tensorflow::DataType tf_dt,
                                               ngraph::element::Type* ng_et);

            // Converts a TensorFlow TensorShape to an nGraph Shape. Requires that none of
            // the dimension lengths in tf_shape are negative.
            bool TFTensorShapeToNGraphShape(const tensorflow::TensorShapeProto& tf_shape,
                                            ngraph::Shape* ng_shape);

            std::shared_ptr<ngraph::Node> GetInputNode(const NodeMap& all_ng_nodes,
                                                       const tensorflow::NodeDef& node,
                                                       size_t input_idx);
            TensorId ParseTensorName(const std::string& name);

            template <typename T, typename VecT = T>
            std::vector<VecT>
                GetValueFromConstOp(std::shared_ptr<ngraph::op::Constant> ng_constant_op)
            {
                // the data type of ngraph::shape is size_t
                std::vector<VecT> dst_values;
                std::vector<T> values = ng_constant_op->get_vector<T>();
                dst_values.resize(values.size());
                
                for (size_t i = 0; i < values.size(); i++)
                {
                    dst_values[i] = static_cast<VecT>(values[i]);
                }
                return dst_values;
            }

            template <typename T>
            bool GetValueFromNGraphOp(std::shared_ptr<ngraph::Node> ng_op, std::vector<T>* values)
            {
                assert(ng_op->description() == "Constant");
                auto ng_constant_op = std::dynamic_pointer_cast<ngraph::op::Constant>(ng_op);
                auto ng_element_type = ng_constant_op->get_element_type();
                if (ng_element_type == ngraph::element::f32)
                {
                    *values = GetValueFromConstOp<float, T>(ng_constant_op);
                }
                else if (ng_element_type == ngraph::element::f64)
                {
                    *values = GetValueFromConstOp<double, T>(ng_constant_op);
                }
                else if (ng_element_type == ngraph::element::i8)
                {
                    *values = GetValueFromConstOp<int8, T>(ng_constant_op);
                }
                else if (ng_element_type == ngraph::element::i16)
                {
                    *values = GetValueFromConstOp<int16, T>(ng_constant_op);
                }
                else if (ng_element_type == ngraph::element::i32)
                {
                    *values = GetValueFromConstOp<int32, T>(ng_constant_op);
                }
                else if (ng_element_type == ngraph::element::i64)
                {
                    *values = GetValueFromConstOp<int64, T>(ng_constant_op);
                }
                else if (ng_element_type == ngraph::element::u8)
                {
                    *values = GetValueFromConstOp<uint8, T>(ng_constant_op);
                }
                else if (ng_element_type == ngraph::element::u16)
                {
                    *values = GetValueFromConstOp<uint16, T>(ng_constant_op);
                }
                else if (ng_element_type == ngraph::element::u32)
                {
                    *values = GetValueFromConstOp<uint32, T>(ng_constant_op);
                }
                else if (ng_element_type == ngraph::element::u64)
                {
                    *values = GetValueFromConstOp<uint64, T>(ng_constant_op);
                }
                else if (ng_element_type == ngraph::element::boolean)
                {
                    *values = GetValueFromConstOp<bool, T>(ng_constant_op);
                }
                else
                {
                    return false;
                }
                return true;
            }

// The ... is to allow the caller to inject some value validation code.  Use
// just ; if no additional validation code is needed.
#define DEFINE_GET_ATTR(TYPE, FIELD, ATTR_TYPE, APPEND_OP, CAST, ...)                              \
    inline bool GetNodeAttr(                                                                       \
        const ::google::protobuf::Map<::std::string, ::tensorflow::AttrValue>& attrs,              \
        std::string name,                                                                          \
        TYPE& value)                                                                               \
    {                                                                                              \
        auto attr = attrs.find(name);                                                              \
        if (attr == attrs.end())                                                                   \
            return false;                                                                          \
        const auto& v = attr->second.FIELD() __VA_ARGS__;                                          \
        value = CAST;                                                                              \
        return true;                                                                               \
    }                                                                                              \
    inline bool GetNodeAttr(                                                                       \
        const ::google::protobuf::Map<::std::string, ::tensorflow::AttrValue>& attrs,              \
        std::string name,                                                                          \
        std::vector<TYPE>& value)                                                                  \
    {                                                                                              \
        auto attr = attrs.find(name);                                                              \
        if (attr == attrs.end())                                                                   \
            return false;                                                                          \
        for (const auto& v : attr->second.list().FIELD())                                          \
        {                                                                                          \
            __VA_ARGS__;                                                                           \
            value.APPEND_OP(CAST);                                                                 \
        }                                                                                          \
        return true;                                                                               \
    }

            DEFINE_GET_ATTR(std::string, s, "string", emplace_back, v, ;)
            DEFINE_GET_ATTR(int64, i, "int", emplace_back, v, ;)
            DEFINE_GET_ATTR(int32, i, "int", emplace_back, static_cast<int32>(v), ;)
            DEFINE_GET_ATTR(float, f, "float", emplace_back, v, ;)
            // std::vector<bool> specialization does not have emplace_back until
            // c++14, so we have to use push_back (see
            // http://en.cppreference.com/w/cpp/container/vector/emplace_back)
            DEFINE_GET_ATTR(bool, b, "bool", push_back, v, ;)
            DEFINE_GET_ATTR(tensorflow::DataType,
                            type,
                            "type",
                            emplace_back,
                            static_cast<tensorflow::DataType>(v),
                            ;);
#undef DEFINE_GET_ATTR

            template <size_t a, size_t b, size_t c, size_t d>
            void Reshape(std::shared_ptr<ngraph::Node>& ng_node)
            {
                static_assert(a < 4 && b < 4 && c < 4 && d < 4,
                              "Number of dimensions cannot exceed 4");
                static_assert(a != b && a != c && a != d && b != c && b != d && c != d,
                              "Dimensions indices cannot be equal");
                auto& s = ng_node->get_shape();
                ngraph::Shape reshaped_shape{s[a], s[b], s[c], s[d]};
                // std::cerr << "reshaping " << ngraph::join(s) << " to "
                //                << ngraph::join(reshaped_shape);
                ng_node = std::make_shared<ngraph::op::Reshape>(
                    ng_node, ngraph::AxisVector{a, b, c, d}, reshaped_shape);
            }

            namespace detail
            {
                template <typename T>
                static inline void NhwcToNGraph(const std::vector<T>& src, std::vector<size_t>& dst)
                {
                    dst[0] = src[1];
                    dst[1] = src[2];
                }

                static inline void NhwcToNGraph(std::shared_ptr<ngraph::Node>& ng_node)
                {
                    Reshape<0, 3, 1, 2>(ng_node);
                }

                template <typename T>
                static inline void NchwToNGraph(const std::vector<T>& src, std::vector<size_t>& dst)
                {
                    dst[0] = src[2];
                    dst[1] = src[3];
                }

                template <typename T>
                static inline void NhwcToNchw(const std::vector<T>& src, std::vector<size_t>& dst)
                {
                    dst[0] = src[0];
                    dst[1] = src[3];
                    dst[2] = src[1];
                    dst[3] = src[2];
                }
            }

            static inline void BatchToNGraph(bool is_nhwc, std::shared_ptr<ngraph::Node>& ng_input)
            {
                if (is_nhwc)
                {
                    detail::NhwcToNGraph(ng_input);
                }
            }

            template <typename T>
            static inline void BatchedOpParamToNGraph(bool is_nhwc,
                                                      const std::vector<T>& src,
                                                      std::vector<size_t>& dst)
            {
                if (is_nhwc)
                {
                    detail::NhwcToNGraph(src, dst);
                }
                else
                {
                    detail::NchwToNGraph(src, dst);
                }
            }

            template <typename T>
            static inline void BatchedOpParamReshape(bool is_nhwc,
                                                     const std::vector<T>& src,
                                                     std::vector<size_t>& dst)
            {
                if (is_nhwc)
                {
                    detail::NhwcToNchw(src, dst);
                }
                else
                {
                    dst = src;
                }
            }

            static inline void BatchToTensorflow(bool is_nhwc,
                                                 std::shared_ptr<ngraph::Node>& ng_node)
            {
                if (!is_nhwc)
                {
                    return;
                }
                Reshape<0, 2, 3, 1>(ng_node);
            }

            template <typename T>
            static inline void MakePadding(const std::string& tf_padding_type,
                                           const ngraph::Shape& ng_image_shape,
                                           const ngraph::Shape& ng_kernel_shape,
                                           const ngraph::Strides& ng_strides,
                                           T& ng_padding_below,
                                           T& ng_padding_above)
            {
                if (tf_padding_type == "SAME")
                {
                    for (size_t i = 0; i < 2; i++)
                    {
                        size_t image_size = ng_image_shape[i];
                        size_t filter_shape = ng_kernel_shape[i];
                        size_t filter_stride = ng_strides[i];

                        int64 padding_needed;
                        if (image_size % filter_stride == 0)
                        {
                            padding_needed = filter_shape - filter_stride;
                        }
                        else
                        {
                            padding_needed = filter_shape - (image_size % filter_stride);
                        }
                        if (padding_needed < 0)
                        {
                            padding_needed = 0;
                        }

                        size_t padding_lhs = padding_needed / 2;
                        size_t padding_rhs = padding_needed - padding_lhs;
                        ng_padding_below[i] = padding_lhs;
                        ng_padding_above[i] = padding_rhs;
                    }
                }

                // std::cerr << "ng_padding_below: " << ngraph::join(ng_padding_below);
                // std::cerr << "ng_padding_above: " << ngraph::join(ng_padding_above);
            }

            template <typename T>
            static inline void MakePadding(const std::string& tf_padding_type,
                                           const ngraph::Shape& ng_image_shape,
                                           const ngraph::Shape& ng_kernel_shape,
                                           const ngraph::Strides& ng_strides,
                                           const ngraph::Shape& ng_dilations,
                                           T& ng_padding_below,
                                           T& ng_padding_above)
            {
                ngraph::Shape ng_dilation_kernel_shape{
                    (ng_kernel_shape[0] - 1) * ng_dilations[0] + 1,
                    (ng_kernel_shape[1] - 1) * ng_dilations[1] + 1};

                MakePadding(tf_padding_type,
                            ng_image_shape,
                            ng_dilation_kernel_shape,
                            ng_strides,
                            ng_padding_below,
                            ng_padding_above);
            }

            static inline bool CheckAxisDimInRange(std::vector<int64> axes, size_t rank)
            {
                for (auto i : axes)
                {
                    if (i < (int)-rank || i >= (int)rank)
                    {
                        std::cerr << "Axis Dimension is out of range. Got " << i
                                  << ", should be in range [-" << rank << ", " << rank << ")";
                        return false;
                    }
                }
                return true;
            }

        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace ngraph
