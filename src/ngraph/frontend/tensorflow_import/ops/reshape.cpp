//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "reshape.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
#include "stdint.h"

namespace ngraph
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            template <typename T>
            ngraph::Shape GetNGraphShape(std::shared_ptr<ngraph::op::Constant> ng_shape_constant,
                                         std::shared_ptr<ngraph::Node> ng_input)
            {
                // the data type of ngraph::shape is size_t
                std::vector<T> shape = ng_shape_constant->get_vector<T>();

                size_t output_rank = shape.size();
                size_t num_input_elements = ngraph::shape_size(ng_input->get_shape());

                // If there is a single "-1" in the result shape, we have to auto-infer
                // the length of that dimension.
                size_t inferred_pos;
                size_t product_of_rest = 1;
                bool seen_inferred = false;
                for (size_t i = 0; i < output_rank; i++)
                {
                    if (shape[i] == -1)
                    {
                        assert(seen_inferred == false);
                        //if (seen_inferred)
                        //{
                        //    return errors::InvalidArgument("Multiple -1 dimensions in result shape");
                        //}
                        inferred_pos = i;
                        seen_inferred = true;
                    }
                    else
                    {
                        product_of_rest *= shape[i];
                    }
                }
                if (seen_inferred)
                {
                    /*
                    if (num_input_elements % product_of_rest != 0)
                    {
                        NGRAPH_VLOG(3) << "{" << ng::join(ng_input->get_shape()) << "}";
                        NGRAPH_VLOG(3) << "{" << ng::join(shape) << "}";
                        return errors::InvalidArgument(
                            "Product of known dimensions (", product_of_rest,
                            ") does not evenly divide the number of input elements (",
                            num_input_elements, ")");
                    }
                    */
                    assert(num_input_elements % product_of_rest == 0);
                    shape[inferred_pos] = num_input_elements / product_of_rest;
                }

                // Convert the values from the constant into an nGraph::Shape, and
                // construct the axis order while we are at it.
                ngraph::Shape ng_shape(output_rank);

                for (size_t i = 0; i < output_rank; i++)
                {
                    ng_shape[i] = shape[i];
                }
                return ng_shape;
            }

            const std::map<
                ngraph::element::Type,
                std::function<ngraph::Shape(std::shared_ptr<ngraph::op::Constant> ng_shape_constant,
                                            std::shared_ptr<ngraph::Node> ng_input)>>&
                NGRAPH_DATATYPE_CONST_MAP()
            {
                static const std::map<ngraph::element::Type,
                                      std::function<ngraph::Shape(
                                          std::shared_ptr<ngraph::op::Constant> ng_shape_constant,
                                          std::shared_ptr<ngraph::Node> ng_input)>>
                    the_map = {{ngraph::element::f32, GetNGraphShape<float>},
                               {ngraph::element::f64, GetNGraphShape<double>},
                               {ngraph::element::i8, GetNGraphShape<int8>},
                               {ngraph::element::i16, GetNGraphShape<int16>},
                               {ngraph::element::i32, GetNGraphShape<int32>},
                               {ngraph::element::i64, GetNGraphShape<int64>},
                               {ngraph::element::u8, GetNGraphShape<uint8>},
                               {ngraph::element::u16, GetNGraphShape<uint16>},
                               {ngraph::element::u32, GetNGraphShape<uint32>},
                               {ngraph::element::u64, GetNGraphShape<uint64>},
                               {ngraph::element::boolean, GetNGraphShape<bool>}};
                return the_map;
            }

            NamedNodeVector TranslateReshapeOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = all_ng_nodes.at(node.input(0));
                auto ng_shape_op = all_ng_nodes.at(node.input(1));

                // TODO: if the shape is placeholder
                auto ng_shape_constant =
                    std::dynamic_pointer_cast<ngraph::op::Constant>(ng_shape_op);
                auto ng_shape_type = ng_shape_op->get_element_type();
                const auto& func_param = NGRAPH_DATATYPE_CONST_MAP().at(ng_shape_type);
                ngraph::Shape ng_shape = func_param(ng_shape_constant, ng_input);

                ngraph::AxisVector ng_axis_order(ng_input->get_shape().size());
                std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                auto ng_node =
                    std::make_shared<ngraph::op::Reshape>(ng_input, ng_axis_order, ng_shape);

                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }
        } // namespace tensorflow_import

    } // namespace frontend

} // namespace ngraph
