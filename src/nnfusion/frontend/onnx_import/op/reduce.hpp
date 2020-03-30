//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <memory>

#include "ngraph/node_vector.hpp"
#include "nnfusion/core/operators/abs.hpp"
#include "nnfusion/core/operators/exp.hpp"
#include "nnfusion/core/operators/log.hpp"
#include "nnfusion/core/operators/max.hpp"
#include "nnfusion/core/operators/min.hpp"
#include "nnfusion/core/operators/multiply.hpp"
#include "nnfusion/core/operators/product.hpp"
#include "nnfusion/core/operators/sqrt.hpp"
#include "nnfusion/core/operators/sum.hpp"

#include "core/node.hpp"
#include "utils/broadcasting.hpp"
#include "utils/reduction.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                /// \brief      Compute the log sum of the input tensor's elements along the provided axes.
                ///
                /// \par Overview
                ///     The output tensor has the same rank as the input if Node attribute keepdims
                ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
                ///     dimension pruned.
                ///
                /// \param[in]  node  The ONNX node representing operation.
                ///
                /// \return     The nGraph node equivalent of the ONNX operation.
                ///
                inline NodeVector reduce_log_sum(const Node& node)
                {
                    auto sum_node = reduction::make_ng_reduction_op<ngraph::op::Sum>(
                        node, node.get_ng_inputs().at(0));
                    return {std::make_shared<ngraph::op::Log>(sum_node)};
                }

                /// \brief      Compute the log sum exponent of the input tensor's elements along the provided axes.
                ///
                /// \par Overview
                ///     The output tensor has the same rank as the input if Node attribute keepdims
                ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
                ///     dimension pruned.
                ///
                /// \param[in]  node  The ONNX node representing operation.
                ///
                /// \return     The nGraph node equivalent of the ONNX operation.
                ///
                inline NodeVector reduce_log_sum_exp(const Node& node)
                {
                    auto exp_node = std::make_shared<ngraph::op::Exp>(node.get_ng_inputs().at(0));
                    auto sum_node =
                        reduction::make_ng_reduction_op<ngraph::op::Sum>(node, exp_node);
                    return {std::make_shared<ngraph::op::Log>(sum_node)};
                }

                /// \brief      Compute the L1 norm of the input tensor's element along the provided axes.
                ///
                /// \par Overview
                ///     The output tensor has the same rank as the input if Node attribute keepdims
                ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
                ///     dimension pruned.
                ///
                /// \param[in]  node  The ONNX node representing operation.
                ///
                /// \return     The nGraph node equivalent of the ONNX operation.
                ///
                inline NodeVector reduce_l1(const Node& node)
                {
                    auto abs_node = std::make_shared<ngraph::op::Abs>(node.get_ng_inputs().at(0));
                    return {reduction::make_ng_reduction_op<ngraph::op::Sum>(node, abs_node)};
                }

                /// \brief      Compute the L2 norm of the input tensor's element along the provided axes.
                ///
                /// \par Overview
                ///     The output tensor has the same rank as the input if Node attribute keepdims
                ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
                ///     dimension pruned.
                ///
                /// \param[in]  node  The ONNX node representing operation.
                ///
                /// \return     The nGraph node equivalent of the ONNX operation.
                ///
                inline NodeVector reduce_l2(const Node& node)
                {
                    NodeVector ng_inputs{node.get_ng_inputs()};
                    auto square_node =
                        std::make_shared<ngraph::op::Multiply>(ng_inputs.at(0), ng_inputs.at(0));
                    auto sum_node =
                        reduction::make_ng_reduction_op<ngraph::op::Sum>(node, square_node);
                    return {std::make_shared<ngraph::op::Sqrt>(sum_node)};
                }

                /// \brief      Compute the maximum value of the input tensor's elements along the provided axes.
                ///
                /// \par Overview
                ///     The output tensor has the same rank as the input if Node attribute keepdims
                ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
                ///     dimension pruned.
                ///
                /// \param[in]  node  The ONNX node representing operation.
                ///
                /// \return     The nGraph node equivalent of the ONNX operation.
                ///
                inline NodeVector reduce_max(const Node& node)
                {
                    return {reduction::make_ng_reduction_op<ngraph::op::Max>(
                        node, node.get_ng_inputs().at(0))};
                }

                /// \brief      Compute the mean value of the input tensor's elements along the provided axes.
                ///
                /// \par Overview
                ///     The output tensor has the same rank as the input if Node attribute keepdims
                ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
                ///     dimension pruned.
                ///
                /// \param[in]  node  The ONNX node representing operation.
                ///
                /// \return     The nGraph node equivalent of the ONNX operation.
                ///
                NodeVector reduce_mean(const Node& node);

                /// \brief      Compute the minimum value of the input tensor's elements along the provided axes.
                ///
                /// \par Overview
                ///     The output tensor has the same rank as the input if Node attribute keepdims
                ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
                ///     dimension pruned.
                ///
                /// \param[in]  node  The ONNX node representing operation.
                ///
                /// \return     The nGraph node equivalent of the ONNX operation.
                ///
                inline NodeVector reduce_min(const Node& node)
                {
                    return {reduction::make_ng_reduction_op<ngraph::op::Min>(
                        node, node.get_ng_inputs().at(0))};
                }

                /// \brief      Compute the product of the input tensor's elements along the provided axes.
                ///
                /// \par Overview
                ///     The output tensor has the same rank as the input if Node attribute keepdims
                ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
                ///     dimension pruned.
                ///
                /// \param[in]  node  The ONNX node representing operation.
                ///
                /// \return     The nGraph node equivalent of the ONNX operation.
                ///
                inline NodeVector reduce_prod(const Node& node)
                {
                    return {reduction::make_ng_reduction_op<ngraph::op::Product>(
                        node, node.get_ng_inputs().at(0))};
                }

                /// \brief      Compute the sum of the input tensor's elements along the provided axes.
                ///
                /// \par Overview
                ///     The output tensor has the same rank as the input if Node attribute keepdims
                ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
                ///     dimension pruned.
                ///
                /// \param[in]  node  The ONNX node representing operation.
                ///
                /// \return     The nGraph node equivalent of the ONNX operation.
                ///
                inline NodeVector reduce_sum(const Node& node)
                {
                    return {reduction::make_ng_reduction_op<ngraph::op::Sum>(
                        node, node.get_ng_inputs().at(0))};
                }

                /// \brief      Compute the sum square of the input tensor's element along the provided axes.
                ///
                /// \par Overview
                ///     The output tensor has the same rank as the input if Node attribute keepdims
                ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
                ///     dimension pruned.
                ///
                /// \param[in]  node  The ONNX node representing operation.
                ///
                /// \return     The nGraph node equivalent of the ONNX operation.
                ///
                inline NodeVector reduce_sum_square(const Node& node)
                {
                    NodeVector ng_inputs{node.get_ng_inputs()};
                    auto square_node =
                        std::make_shared<ngraph::op::Multiply>(ng_inputs.at(0), ng_inputs.at(0));
                    return {reduction::make_ng_reduction_op<ngraph::op::Sum>(node, square_node)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
