//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <algorithm>
#include <memory>
#include <vector>

#include "ngraph/node.hpp"
#include "nnfusion/core/operators/slice.hpp"

#include "slice.hpp"
#include "utils/common.hpp"

static inline int64_t get_valid_array_idx(int64_t idx, int64_t last_idx)
{
    return (idx >= 0) ? std::min(idx, last_idx) : std::max<int64_t>(0, last_idx + idx);
}

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector slice(const Node& node)
                {
                    std::shared_ptr<ngraph::Node> data = node.get_ng_inputs().at(0);
                    Shape data_shape = data->get_shape();

                    auto starts = node.get_attribute_value<std::vector<int64_t>>("starts");
                    auto ends = node.get_attribute_value<std::vector<int64_t>>("ends");

                    auto axes = node.get_attribute_value<std::vector<int64_t>>(
                        "axes", common::get_monotonic_range<int64_t>(data_shape.size()));

                    Shape lower_bounds(data_shape.size());
                    Shape upper_bounds = data_shape;

                    for (auto idx = 0; idx < axes.size(); ++idx)
                    {
                        size_t axis = axes.at(idx);
                        lower_bounds.at(axis) =
                            get_valid_array_idx(starts.at(idx), data_shape.at(axis));
                        upper_bounds.at(axis) =
                            get_valid_array_idx(ends.at(idx), data_shape.at(axis));
                    }

                    return {std::make_shared<ngraph::op::Slice>(data, lower_bounds, upper_bounds)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
