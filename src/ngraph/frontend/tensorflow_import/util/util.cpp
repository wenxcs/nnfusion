//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "util.hpp"
namespace ngraph
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            bool TFDataTypeToNGraphElementType(const tensorflow::DataType tf_dt,
                                               ngraph::element::Type* ng_et)
            {
                switch (tf_dt)
                {
                case tensorflow::DataType::DT_FLOAT: *ng_et = ngraph::element::f32; break;
                case tensorflow::DataType::DT_DOUBLE: *ng_et = ngraph::element::f64; break;
                case tensorflow::DataType::DT_INT8: *ng_et = ngraph::element::i8; break;
                case tensorflow::DataType::DT_INT16: *ng_et = ngraph::element::i16; break;
                case tensorflow::DataType::DT_INT32: *ng_et = ngraph::element::i32; break;
                case tensorflow::DataType::DT_INT64: *ng_et = ngraph::element::i64; break;
                case tensorflow::DataType::DT_UINT8: *ng_et = ngraph::element::u8; break;
                case tensorflow::DataType::DT_UINT16: *ng_et = ngraph::element::u16; break;
                case tensorflow::DataType::DT_UINT32: *ng_et = ngraph::element::u32; break;
                case tensorflow::DataType::DT_UINT64: *ng_et = ngraph::element::u64; break;
                case tensorflow::DataType::DT_BOOL: *ng_et = ngraph::element::boolean; break;
                case tensorflow::DataType::DT_QINT8: *ng_et = ngraph::element::i8; break;
                case tensorflow::DataType::DT_QUINT8: *ng_et = ngraph::element::u8; break;
                default: return false;
                }
                return true;
            }

            bool TFTensorShapeToNGraphShape(const tensorflow::TensorShapeProto& tf_shape,
                                            ngraph::Shape* ng_shape)
            {
                for (int i = 0; i < tf_shape.dim_size(); i++)
                {
                    if (tf_shape.dim(i).size() < 0)
                    {
                        return false;
                    }
                }

                *ng_shape = ngraph::Shape(tf_shape.dim_size());
                for (int i = 0; i < tf_shape.dim_size(); i++)
                {
                    (*ng_shape)[i] = tf_shape.dim(i).size();
                }

                return true;
            }
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace ngraph
