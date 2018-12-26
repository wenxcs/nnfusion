//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "tensorflow_base.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            bool GetNodeAttr(
                const ::google::protobuf::Map<::std::string, ::tensorflow::AttrValue>& attrs,
                std::string name,
                tensorflow::DataType& data_type)
            {
                auto attr = attrs.find(name);
                if (attr == attrs.end())
                    return false;
                data_type = attr->second.type();
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