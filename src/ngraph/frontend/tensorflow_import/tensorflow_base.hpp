//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "ngraph/frontend/base.hpp"
#include "ngraph/function.hpp"
#include "proto/graph.pb.h"

namespace ngraph
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            using NamedNode = std::pair<std::string, std::shared_ptr<ngraph::Node>>;
            using NamedNodeVector = std::vector<NamedNode>;
            using NodeMap = std::map<std::string, std::shared_ptr<ngraph::Node>>;
            using ConvertFunc = std::function<NamedNodeVector(const tensorflow::NodeDef&, const NodeMap&)>;

            inline void CopyToArray(const std::string& src, char* dst)
            {
                memcpy(dst, src.data(), src.size());
            }

            // Move to utils
            bool GetNodeAttr(
                const ::google::protobuf::Map<::std::string, ::tensorflow::AttrValue>& attrs,
                std::string name,
                tensorflow::DataType& data_type);

            // Move to utils
            bool TFTensorShapeToNGraphShape(const tensorflow::TensorShapeProto& tf_shape,
                                            ngraph::Shape* ng_shape);
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace ngraph