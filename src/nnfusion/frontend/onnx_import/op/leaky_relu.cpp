//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/shape.hpp"

#include "nnfusion/core/operators/broadcast.hpp"
#include "nnfusion/core/operators/constant.hpp"
#include "nnfusion/core/operators/maximum.hpp"
#include "nnfusion/core/operators/multiply.hpp"

#include "exceptions.hpp"

#include "core/node.hpp"
#include "utils/broadcasting.hpp"

#include "leaky_relu.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector leaky_relu(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    double alpha = node.get_attribute_value<double>("alpha", 0.01);

                    ASSERT_VALID_ARGUMENT(node, ((alpha >= 0) && (alpha <= 1)))
                        << " alpha value should be in range (0,1)";

                    std::shared_ptr<ngraph::Node> alpha_node =
                        std::make_shared<ngraph::op::Constant>(
                            data->get_element_type(), Shape{}, std::vector<double>{alpha});
                    alpha_node = make_broadcast_node(alpha_node, data->get_shape());
                    return {std::make_shared<ngraph::op::Maximum>(data * alpha_node, data)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
