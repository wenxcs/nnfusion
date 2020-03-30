//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <numeric>

#include "nnfusion/core/operators/softmax.hpp"

#include "exceptions.hpp"
#include "softmax.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector softmax(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    auto data = inputs.at(0);
                    auto data_shape = data->get_shape();

                    int axis = node.get_attribute_value<int64_t>("axis", 1);

                    if (axis < 0)
                    {
                        axis = data_shape.size() + axis;
                    }

                    ASSERT_VALID_ARGUMENT(node, axis < data_shape.size())
                        << "provided 'axis' value:" << axis
                        << " is out of input tensor dimensions range.";

                    // create vector of capacity data_dimensions - axis_divider position
                    std::vector<size_t> axes(data_shape.size() - axis);
                    std::iota(std::begin(axes), std::end(axes), axis);
                    return {std::make_shared<ngraph::op::Softmax>(data, axes)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
