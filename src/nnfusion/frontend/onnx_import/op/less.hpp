//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "ngraph/node_vector.hpp"
#include "nnfusion/core/operators/less.hpp"

#include "core/node.hpp"
#include "utils/broadcasting.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline NodeVector less(const Node& node)
                {
                    NodeVector ng_inputs{
                        numpy_style_broadcast_for_binary_operation(node.get_ng_inputs())};
                    return {std::make_shared<ngraph::op::Less>(ng_inputs.at(0), ng_inputs.at(1))};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
