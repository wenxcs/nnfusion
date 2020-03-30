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
                /// \brief Convert ONNX GlobalMaxPool operation to an nGraph node.
                ///
                /// \param node   The ONNX node object representing this operation.
                ///
                /// \return The vector containing Ngraph nodes producing output of ONNX GlobalMaxPool
                ///         operation.
                NodeVector global_max_pool(const Node& node);

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
