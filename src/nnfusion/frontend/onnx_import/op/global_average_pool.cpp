//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "nnfusion/core/operators/avg_pool.hpp"

#include "utils/convpool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector global_average_pool(const Node& node)
                {
                    return convpool::make_ng_pool<ngraph::op::AvgPool>(node);
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
