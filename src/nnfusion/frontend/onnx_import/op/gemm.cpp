//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "op/gemm.hpp"

#include "nnfusion/core/operators/add.hpp"
#include "nnfusion/core/operators/broadcast.hpp"
#include "nnfusion/core/operators/constant.hpp"
#include "nnfusion/core/operators/dot.hpp"
#include "nnfusion/core/operators/multiply.hpp"

#include "../exceptions.hpp"
#include "../utils/broadcasting.hpp"
#include "../utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector gemm(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    auto input_a = inputs.at(0);
                    auto input_b = inputs.at(1);
                    auto input_c = inputs.at(2);

                    double alpha = node.get_attribute_value<double>("alpha", 1);
                    double beta = node.get_attribute_value<double>("beta", 1);

                    auto trans_a = node.get_attribute_value<int64_t>("transA", 0);
                    auto trans_b = node.get_attribute_value<int64_t>("transB", 0);

                    if (trans_a != 0)
                    {
                        input_a = reshape::transpose(input_a);
                    }
                    if (trans_b != 0)
                    {
                        input_b = reshape::transpose(input_b);
                    }

                    // code from python not implemented in c++ yet.
                    // reshape_for_matmul(node, input_a, input_b);

                    // A' * B'
                    std::shared_ptr<ngraph::Node> a_dot_b =
                        std::make_shared<ngraph::op::Dot>(input_a, input_b);

                    // alpha
                    std::shared_ptr<ngraph::Node> alpha_node =
                        std::make_shared<ngraph::op::Constant>(a_dot_b->get_element_type(),
                                                               ngraph::Shape{},
                                                               std::vector<double>{alpha});
                    alpha_node = make_broadcast_node(alpha_node, a_dot_b->get_shape());
                    // alpha * A' * B'
                    a_dot_b = std::make_shared<ngraph::op::Multiply>(alpha_node, a_dot_b);

                    // beta * C
                    std::shared_ptr<ngraph::Node> beta_node =
                        std::make_shared<ngraph::op::Constant>(input_c->get_element_type(),
                                                               ngraph::Shape{},
                                                               std::vector<double>{beta});
                    beta_node = make_broadcast_node(beta_node, input_c->get_shape());
                    input_c = std::make_shared<ngraph::op::Multiply>(beta_node, input_c);

                    // alpha * A' * B' + beta * C
                    NodeVector broadcasted_nodes =
                        numpy_style_broadcast_for_binary_operation(a_dot_b, input_c);
                    // The ONNX documentation says that `input_c` should be "unidirectional broadcastable"
                    // to the `a_dot_b` tensor. Since numpy style broadcasting is bidirectional, below we
                    // only use the second output from above broadcasting. In other words we want to
                    // preserve the shape of original `a_dot_b` tensor.
                    return {std::make_shared<ngraph::op::Add>(a_dot_b, broadcasted_nodes.at(1))};
                }

            } // namespace set_1

        } //namespace op

    } // namespace  onnx_import

} // namespace  ngraph
