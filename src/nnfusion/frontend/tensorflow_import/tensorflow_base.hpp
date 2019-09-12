//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "graph.pb.h"
#include "ngraph/function.hpp"
#include "ngraph/op/parameter_vector.hpp"
#include "nnfusion/frontend/base.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            using NamedNode = std::pair<std::string, std::shared_ptr<ngraph::Node>>;
            using NamedNodeVector = std::vector<NamedNode>;
            using NodeMap = std::map<std::string, std::vector<std::shared_ptr<ngraph::Node>>>;
            using ConvertFunc = std::function<NamedNodeVector(
                const tensorflow::NodeDef&, const NodeMap&, ngraph::op::ParameterVector&)>;

            typedef signed char int8;
            typedef short int16;
            typedef int int32;
            typedef long long int64;

            typedef unsigned char uint8;
            typedef unsigned short uint16;
            typedef unsigned int uint32;
            typedef unsigned long long uint64;

            inline void CopyToArray(const std::string& src, char* dst)
            {
                memcpy(dst, src.data(), src.size());
            }
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace nnfusion