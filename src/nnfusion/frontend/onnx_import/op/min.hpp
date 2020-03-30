//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "ngraph/node_vector.hpp"
#include "nnfusion/core/operators/minimum.hpp"

#include "core/node.hpp"
#include "utils/variadic.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline NodeVector min(const Node& node)
                {
                    return variadic::make_ng_variadic_op<ngraph::op::Minimum>(node);
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
