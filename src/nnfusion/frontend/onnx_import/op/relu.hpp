//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "ngraph/node_vector.hpp"
#include "nnfusion/core/operators/relu.hpp"

#include "core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline NodeVector relu(const Node& node)
                {
                    NodeVector ng_inputs{node.get_ng_inputs()};
                    return {std::make_shared<ngraph::op::Relu>(ng_inputs.at(0))};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
