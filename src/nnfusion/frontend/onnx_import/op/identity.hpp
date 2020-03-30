//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "ngraph/node_vector.hpp"

#include "core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline NodeVector identity(const Node& node)
                {
                    return {node.get_ng_inputs().at(0)};
                }
            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
