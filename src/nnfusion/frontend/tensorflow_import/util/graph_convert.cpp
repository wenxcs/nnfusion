//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "graph_convert.hpp"
#include "../ops/const.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/numpy_transpose.hpp"
#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "util/bcast.hpp"

#include "nnfusion/core/ops/generic_op.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            // Using this policy if no explict tf_import mapping exists
            NamedNodeVector TranslateGenericNoAttrOp(const tensorflow::NodeDef& node,
                                                     const NodeMap& all_ng_nodes,
                                                     ngraph::op::ParameterVector& parameters)
            {
                std::vector<std::shared_ptr<Node>> inputs;
                size_t input_cnt = node.input_size();
                for (int i = 0; i < input_cnt; i++)
                    inputs.push_back(GetInputNode(all_ng_nodes, node, i));

                auto out_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(),
                    node.op(), // select which existing kernels to use;
                    inputs,
                    ngraph::op::OpConfig::any{});
                NamedNodeVector ret{{node.name(), out_node}};
                return ret;
            }

            NamedNodeVector TranslateIdentityOp(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                ngraph::op::ParameterVector& parameters)
            {
                auto input_node = GetInputNode(all_ng_nodes, node, 0);
                NamedNodeVector ret{{node.name(), input_node}};
                return ret;
            }

            NamedNodeVector TranslateInvertPermutationOp(const tensorflow::NodeDef& node,
                                                         const NodeMap& all_ng_nodes,
                                                         ngraph::op::ParameterVector& parameters)
            {
                auto inputs = GetAllInputNode(all_ng_nodes, node);
                ngraph::op::OpConfig::any myConfig;

                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(), node.op(), inputs, myConfig);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};

                return ret;
            }

            NamedNodeVector TranslateNoOp(const tensorflow::NodeDef& node,
                                          const NodeMap& all_ng_nodes,
                                          ngraph::op::ParameterVector& parameters)
            {
                NamedNodeVector ret;
                size_t input_cnt = node.input_size();
                for (int i = 0; i < input_cnt; i++)
                {
                    TensorId input_tensor(ParseTensorName(node.input(i)));
                    if (input_tensor.second >= 0)
                    {
                        auto input_node = GetInputNode(all_ng_nodes, node, i);
                        ret.push_back({node.name(), input_node});
                    }
                }
                return ret;
            }

            template <typename T>
            NamedNodeVector TranslateUnaryOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             ngraph::op::ParameterVector& parameters)
            {
                auto input_node = GetInputNode(all_ng_nodes, node, 0);
                auto ng_node = std::make_shared<T>(input_node);
                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            template <typename T>
            NamedNodeVector TranslateBinaryOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              ngraph::op::ParameterVector& parameters)
            {
                auto ng_lhs = GetInputNode(all_ng_nodes, node, 0);
                auto ng_rhs = GetInputNode(all_ng_nodes, node, 1);
                std::tie(ng_lhs, ng_rhs) =
                    ngraph::builder::numpy_broadcast(std::make_pair(ng_lhs, ng_rhs));
                auto ng_node = std::make_shared<T>(ng_lhs, ng_rhs);
                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            template <typename T>
            NamedNodeVector TranslateInputOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             ngraph::op::ParameterVector& parameters)
            {
                tensorflow::DataType dtype;
                assert(GetNodeAttr(node.attr(), "dtype", dtype) == true);
                ngraph::element::Type ng_et;
                assert(TFDataTypeToNGraphElementType(dtype, &ng_et) == true);
                tensorflow::TensorShapeProto tf_shape = node.attr().at("shape").shape();
                ngraph::Shape ng_shape;
                assert(TFTensorShapeToNGraphShape(tf_shape, &ng_shape));

                auto ng_node = std::make_shared<T>(ng_et, ng_shape);
                ng_node->set_name(node.name());
                parameters.push_back(ng_node);
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateSparseSoftmaxCrossEntropyWithLogitsOp(
                const tensorflow::NodeDef& node,
                const NodeMap& all_ng_nodes,
                ngraph::op::ParameterVector& parameters)
            {
                auto ng_lhs = GetInputNode(all_ng_nodes, node, 0);
                auto ng_rhs = GetInputNode(all_ng_nodes, node, 1);

                ngraph::AxisSet ng_axes_softmax{ng_lhs->get_shape().size() - 1};
                auto ng_softmax = std::make_shared<ngraph::op::Softmax>(ng_lhs, ng_axes_softmax);

                auto loss_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(),
                    "CrossEntropyAvgLossWithLabels", // select which existing kernels to use;
                    std::vector<std::shared_ptr<Node>>({ng_softmax, ng_rhs}),
                    ngraph::op::OpConfig::any{});

                auto bwd_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(),
                    "CrossEntropyFwdBwdWithSoftmaxBwd", // select which existing kernels to use;
                    std::vector<std::shared_ptr<Node>>({ng_softmax, ng_rhs}),
                    ngraph::op::OpConfig::any{});

                loss_node->set_name(node.name());
                bwd_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), loss_node}, {node.name(), bwd_node}};
                return ret;
            }

            NamedNodeVector TranslateMatMulOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              ngraph::op::ParameterVector& parameters)
            {
                auto ng_lhs = GetInputNode(all_ng_nodes, node, 0);
                auto ng_rhs = GetInputNode(all_ng_nodes, node, 1);
                // Transpose arguments if requested.
                bool transpose_a = false;
                bool transpose_b = false;
                assert(GetNodeAttr(node.attr(), "transpose_a", transpose_a) == true);
                assert(GetNodeAttr(node.attr(), "transpose_b", transpose_b) == true);
                // if (transpose_a)
                // {
                //     ng_lhs = ngraph::builder::numpy_transpose(ng_lhs, ngraph::AxisVector{1, 0});
                // }
                // if (transpose_b)
                // {
                //     ng_rhs = ngraph::builder::numpy_transpose(ng_rhs, ngraph::AxisVector{1, 0});
                // }

                auto ng_node = std::make_shared<ngraph::op::Dot>(
                    ng_lhs, ng_rhs, 0, false, transpose_a, transpose_b);
                ng_node->set_name(node.name());
                //ng_node->set_transpose(transpose_a, transpose_b);
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateBatchMatMulOp(const tensorflow::NodeDef& node,
                                                   const NodeMap& all_ng_nodes,
                                                   ngraph::op::ParameterVector& parameters)
            {
                auto ng_lhs = GetInputNode(all_ng_nodes, node, 0);
                auto ng_rhs = GetInputNode(all_ng_nodes, node, 1);
                // Transpose arguments if requested.
                bool adj_x = false;
                bool adj_y = false;

                assert(GetNodeAttr(node.attr(), "adj_x", adj_x) == true);
                assert(GetNodeAttr(node.attr(), "adj_y", adj_y) == true);

                int input_dims = ng_lhs->get_output_shape(0).size();
                ngraph::AxisVector ng_axis_order;

                ng_axis_order.reserve(input_dims);

                for (int i = 0; i < input_dims - 2; i++)
                {
                    ng_axis_order.push_back(i);
                }
                ng_axis_order.push_back(input_dims - 1);
                ng_axis_order.push_back(input_dims - 2);

                // if (adj_x)
                // {
                //     ng_lhs = ngraph::builder::numpy_transpose(ng_lhs, ng_axis_order);
                // }
                // if (adj_y)
                // {
                //     ng_rhs = ngraph::builder::numpy_transpose(ng_rhs, ng_axis_order);
                // }

                ngraph::op::OpConfig::any myConfig;
                myConfig["adj_x"]["b"] = adj_x;
                myConfig["adj_y"]["b"] = adj_y;

                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(),
                    "BatchMatMul", // select which existing kernels to use;
                    std::vector<std::shared_ptr<Node>>({ng_lhs, ng_rhs}),
                    myConfig);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateBiasAddOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_bias = GetInputNode(all_ng_nodes, node, 1);
                std::string tf_data_format;
                assert(GetNodeAttr(node.attr(), "data_format", tf_data_format) == true);

                if (tf_data_format != "NHWC" && tf_data_format != "NCHW")
                {
                    std::cerr << "BiasAdd data format is neither NHWC nor NCHW";
                    assert(false);
                }

                auto ng_input_shape = ng_input->get_shape();
                auto ng_bias_shape = ng_bias->get_shape();

                if (ng_bias_shape.size() != 1)
                {
                    std::cerr << "Bias argument to BiasAdd does not have one dimension";
                    assert(false);
                }

                bool is_nhwc = (tf_data_format == "NHWC");

                ngraph::AxisSet ng_broadcast_axes;

                if (is_nhwc)
                {
                    for (size_t i = 0; i < ng_input_shape.size() - 1; i++)
                    {
                        ng_broadcast_axes.insert(i);
                    }
                }
                else
                {
                    for (size_t i = 0; i < ng_input_shape.size(); i++)
                    {
                        if (i != 1)
                        {
                            ng_broadcast_axes.insert(i);
                        }
                    }
                }

                auto ng_bias_broadcasted = std::make_shared<ngraph::op::Broadcast>(
                    ng_bias, ng_input_shape, ng_broadcast_axes);

                auto ng_add = ng_input + ng_bias_broadcasted;

                ng_add->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_add}};
                return ret;
            }

            NamedNodeVector TranslateReluGradOp(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                ngraph::op::ParameterVector& parameters)
            {
                auto ng_delta = GetInputNode(all_ng_nodes, node, 0);
                auto ng_arg = GetInputNode(all_ng_nodes, node, 1);
                auto ng_node = std::make_shared<ngraph::op::ReluBackprop>(ng_arg, ng_delta);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateBiasAddGradOp(const tensorflow::NodeDef& node,
                                                   const NodeMap& all_ng_nodes,
                                                   ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                std::string tf_data_format;
                assert(GetNodeAttr(node.attr(), "data_format", tf_data_format) == true);

                if (tf_data_format == "")
                {
                    tf_data_format = "NHWC";
                }

                if (tf_data_format != "NHWC" && tf_data_format != "NCHW")
                {
                    std::cerr << "BiasAddGrad data format is neither NHWC nor NCHW";
                    assert(false);
                }

                auto ng_input_shape = ng_input->get_shape();

                if (ng_input_shape.size() < 2)
                {
                    std::cerr << "Input tensor must be at least 2D";
                    assert(false);
                }

                bool is_nhwc = (tf_data_format == "NHWC");

                ngraph::AxisSet ng_reduction_axes;

                if (is_nhwc)
                {
                    for (size_t i = 0; i < ng_input_shape.size() - 1; i++)
                    {
                        ng_reduction_axes.insert(i);
                    }
                }

                else
                {
                    for (size_t i = 0; i < ng_input_shape.size(); i++)
                    {
                        if (i != 1)
                        {
                            ng_reduction_axes.insert(i);
                        }
                    }
                }

                auto ng_bias_add_grad =
                    std::make_shared<ngraph::op::Sum>(ng_input, ng_reduction_axes);

                ng_bias_add_grad->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_bias_add_grad}};
                return ret;
            }

            NamedNodeVector TranslateReshapeOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_shape_op = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int64> shape;
                assert(GetValueFromNGraphOp<int64>(ng_shape_op, &shape) == true);

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

                ngraph::AxisVector ng_axis_order(ng_input->get_shape().size());
                std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                auto ng_node =
                    std::make_shared<ngraph::op::Reshape>(ng_input, ng_axis_order, ng_shape);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateCastOp(const tensorflow::NodeDef& node,
                                            const NodeMap& all_ng_nodes,
                                            ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                tensorflow::DataType dtype;
                assert(GetNodeAttr(node.attr(), "DstT", dtype) == true);
                ngraph::element::Type ng_et;
                assert(TFDataTypeToNGraphElementType(dtype, &ng_et) == true);
                auto ng_node = std::make_shared<ngraph::op::Convert>(ng_input, ng_et);
                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateMaxPoolOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                std::vector<int32> tf_strides;
                std::vector<int32> tf_ksize;
                std::string tf_padding_type;
                std::string tf_data_format;

                assert(GetNodeAttr(node.attr(), "strides", tf_strides) == true);
                assert(GetNodeAttr(node.attr(), "ksize", tf_ksize) == true);
                assert(GetNodeAttr(node.attr(), "padding", tf_padding_type) == true);
                assert(GetNodeAttr(node.attr(), "data_format", tf_data_format) == true);

                if (tf_data_format != "NHWC" && tf_data_format != "NCHW")
                {
                    std::cerr << "MaxPool data format is neither NHWC nor NCHW";
                    assert(false);
                }

                bool is_nhwc = (tf_data_format == "NHWC");
                ngraph::Strides ng_strides(2);
                ngraph::Shape ng_image_shape(2);
                ngraph::Shape ng_kernel_shape(2);

                BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
                BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
                BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);
                BatchToNGraph(is_nhwc, ng_input);

                // TODO: change this once nGraph supports negative padding
                // (CoordinateDiff) for MaxPool
                // ng::CoordinateDiff ng_padding_below{0,0};
                // ng::CoordinateDiff ng_padding_above{0,0};

                ngraph::Shape ng_padding_below{0, 0};
                ngraph::Shape ng_padding_above{0, 0};
                MakePadding(tf_padding_type,
                            ng_image_shape,
                            ng_kernel_shape,
                            ng_strides,
                            ng_padding_below,
                            ng_padding_above);

                std::shared_ptr<ngraph::Node> ng_maxpool = std::make_shared<ngraph::op::MaxPool>(
                    ng_input, ng_kernel_shape, ng_strides, ng_padding_below, ng_padding_above);

                BatchToTensorflow(is_nhwc, ng_maxpool);
                //std::cerr << "maxpool outshape: {" << ngraph::join(ng_maxpool->get_shape()) << "}";

                ng_maxpool->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_maxpool}};
                return ret;
            }

            NamedNodeVector TranslateConv2DOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              ngraph::op::ParameterVector& parameters)
            {
                // <Todo: wenxh> Group Conv2D
                std::vector<int> tf_strides;
                std::vector<int> tf_dilations;
                std::string tf_padding_type;
                std::string tf_data_format;
                // Make sure the order maters!
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_filter = GetInputNode(all_ng_nodes, node, 1);

                assert(GetNodeAttr(node.attr(), "strides", tf_strides));
                assert(GetNodeAttr(node.attr(), "dilations", tf_dilations));
                assert(GetNodeAttr(node.attr(), "padding", tf_padding_type));
                assert(GetNodeAttr(node.attr(), "data_format", tf_data_format));
                assert(tf_data_format == "NHWC" || tf_data_format == "NCHW");

                bool is_nhwc = (tf_data_format == "NHWC");
                ngraph::Strides ng_strides(2);
                ngraph::Strides ng_dilations(2);
                ngraph::Shape ng_image_shape(2);
                ngraph::Shape ng_kernel_shape(2);

                BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
                BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
                BatchedOpParamToNGraph(is_nhwc, tf_dilations, ng_dilations);
                BatchToNGraph(is_nhwc, ng_input);

                auto& ng_filter_shape = ng_filter->get_shape();
                ng_kernel_shape[0] = ng_filter_shape[0];
                ng_kernel_shape[1] = ng_filter_shape[1];
                Reshape<3, 2, 0, 1>(ng_filter);

                // Padding
                ngraph::CoordinateDiff ng_padding_below{0, 0};
                ngraph::CoordinateDiff ng_padding_above{0, 0};

                MakePadding(tf_padding_type,
                            ng_image_shape,
                            ng_kernel_shape,
                            ng_strides,
                            ng_dilations,
                            ng_padding_below,
                            ng_padding_above);

                // Generate new op
                std::shared_ptr<ngraph::Node> ng_conv =
                    std::make_shared<ngraph::op::Convolution>(ng_input,
                                                              ng_filter,
                                                              ng_strides,
                                                              ng_dilations,
                                                              ng_padding_below,
                                                              ng_padding_above);
                BatchToTensorflow(is_nhwc, ng_conv);
                ng_conv->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_conv}};
                return ret;
            }

            NamedNodeVector TranslateAvgPoolOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                std::vector<int32> tf_strides;
                std::vector<int32> tf_ksize;
                std::string tf_padding_type;
                std::string tf_data_format;

                assert(GetNodeAttr(node.attr(), "strides", tf_strides) == true);
                assert(GetNodeAttr(node.attr(), "ksize", tf_ksize) == true);
                assert(GetNodeAttr(node.attr(), "padding", tf_padding_type) == true);
                assert(GetNodeAttr(node.attr(), "data_format", tf_data_format) == true);

                if (tf_data_format != "NHWC" && tf_data_format != "NCHW")
                {
                    std::cerr << "AvgPool data format is neither NHWC nor NCHW";
                    assert(false);
                }

                bool is_nhwc = (tf_data_format == "NHWC");
                ngraph::Strides ng_strides(2);
                ngraph::Shape ng_image_shape(2);
                ngraph::Shape ng_kernel_shape(2);

                BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
                BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
                BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);
                BatchToNGraph(is_nhwc, ng_input);

                // TODO: change this once nGraph supports negative padding
                // (CoordinateDiff) for AvgPool
                // ng::CoordinateDiff ng_padding_below{0,0};
                // ng::CoordinateDiff ng_padding_above{0,0};

                ngraph::Shape ng_padding_below{0, 0};
                ngraph::Shape ng_padding_above{0, 0};
                MakePadding(tf_padding_type,
                            ng_image_shape,
                            ng_kernel_shape,
                            ng_strides,
                            ng_padding_below,
                            ng_padding_above);

                std::shared_ptr<ngraph::Node> ng_avgpool =
                    std::make_shared<ngraph::op::AvgPool>(ng_input,
                                                          ng_kernel_shape,
                                                          ng_strides,
                                                          ng_padding_below,
                                                          ng_padding_above,
                                                          false);

                BatchToTensorflow(is_nhwc, ng_avgpool);
                //std::cerr << "avgpool outshape: {" << ngraph::join(ng_avgpool->get_shape()) << "}";

                ng_avgpool->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_avgpool}};
                return ret;
            }

            NamedNodeVector TranslateFillOp(const tensorflow::NodeDef& node,
                                            const NodeMap& all_ng_nodes,
                                            ngraph::op::ParameterVector& parameters)
            {
                auto ng_shape_op = GetInputNode(all_ng_nodes, node, 0);
                auto ng_value = GetInputNode(all_ng_nodes, node, 1);

                std::vector<size_t> dims_vec;
                assert(GetValueFromNGraphOp<size_t>(ng_shape_op, &dims_vec) == true);

                ngraph::Shape ng_output_shape(dims_vec.size());
                ngraph::AxisSet ng_axis_set;
                for (size_t i = 0; i < dims_vec.size(); ++i)
                {
                    ng_output_shape[i] = dims_vec[i];
                    ng_axis_set.insert(i);
                }

                std::shared_ptr<ngraph::Node> ng_fill =
                    std::make_shared<ngraph::op::Broadcast>(ng_value, ng_output_shape, ng_axis_set);
                ng_fill->set_name(node.name());

                NamedNodeVector ret{{node.name(), ng_fill}};
                return ret;
            }

            NamedNodeVector TranslatePadOp(const tensorflow::NodeDef& node,
                                           const NodeMap& all_ng_nodes,
                                           ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_padding_op = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int64> paddings;
                assert(GetValueFromNGraphOp<int64>(ng_padding_op, &paddings) == true);

                if (paddings.size() % 2 != 0)
                {
                    std::cerr << "Constant node for paddings does not have an even number of "
                                 "elements";
                    assert(false);
                }

                ngraph::Shape padding_below(paddings.size() / 2);
                ngraph::Shape padding_above(paddings.size() / 2);
                ngraph::Shape padding_interior(paddings.size() / 2);

                for (size_t i = 0; i < paddings.size() / 2; i++)
                {
                    padding_below[i] = paddings[2 * i];
                    padding_above[i] = paddings[2 * i + 1];
                    padding_interior[i] = 0;
                }

                // For PadV1 it seems the value is always zero.
                auto ng_pad_val_op = std::make_shared<ngraph::op::Constant>(
                    ng_input->get_element_type(), ngraph::Shape{}, std::vector<std::string>{"0"});
                auto ng_pad = std::make_shared<ngraph::op::Pad>(
                    ng_input, ng_pad_val_op, padding_below, padding_above, padding_interior);

                ng_pad->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_pad}};
                return ret;
            }

            NamedNodeVector TranslatePadV2Op(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_padding_op = GetInputNode(all_ng_nodes, node, 1);
                auto ng_constant_value_op = GetInputNode(all_ng_nodes, node, 2);

                std::vector<int64> paddings;
                assert(GetValueFromNGraphOp<int64>(ng_padding_op, &paddings) == true);

                assert(ng_constant_value_op->description() == "Constant");
                auto ng_constant_op =
                    std::dynamic_pointer_cast<ngraph::op::Constant>(ng_constant_value_op);
                auto constant_values = ng_constant_op->get_value_strings();

                if (paddings.size() % 2 != 0)
                {
                    std::cerr << "Constant node for paddings does not have an even number of "
                                 "elements";
                    assert(false);
                }

                ngraph::Shape padding_below(paddings.size() / 2);
                ngraph::Shape padding_above(paddings.size() / 2);
                ngraph::Shape padding_interior(paddings.size() / 2);

                for (size_t i = 0; i < paddings.size() / 2; i++)
                {
                    padding_below[i] = paddings[2 * i];
                    padding_above[i] = paddings[2 * i + 1];
                    padding_interior[i] = 0;
                }

                // For PadV1 it seems the value is always zero.
                auto ng_pad_val_op = std::make_shared<ngraph::op::Constant>(
                    ng_input->get_element_type(), ngraph::Shape{}, constant_values);
                auto ng_pad = std::make_shared<ngraph::op::Pad>(
                    ng_input, ng_pad_val_op, padding_below, padding_above, padding_interior);

                ng_pad->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_pad}};
                return ret;
            }

            NamedNodeVector TranslateFusedBatchNormOp(const tensorflow::NodeDef& node,
                                                      const NodeMap& all_ng_nodes,
                                                      ngraph::op::ParameterVector& parameters)
            {
                bool tf_is_training;
                if (GetNodeAttr(node.attr(), "is_training", tf_is_training) == false)
                {
                    std::cout << "is_training attribute not present, setting to true";
                    tf_is_training = true;
                }
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_scale = GetInputNode(all_ng_nodes, node, 1);
                auto ng_offset = GetInputNode(all_ng_nodes, node, 2);
                auto ng_mean = GetInputNode(all_ng_nodes, node, 3);
                auto ng_variance = GetInputNode(all_ng_nodes, node, 4);

                std::string tf_data_format;
                assert(GetNodeAttr(node.attr(), "data_format", tf_data_format));

                if (tf_data_format != "NHWC" && tf_data_format != "NCHW")
                {
                    std::cerr << "FusedBatchNorm data format is neither NHWC nor NCHW";
                    assert(false);
                }

                bool is_nhwc = (tf_data_format == "NHWC");
                float tf_epsilon;
                if (GetNodeAttr(node.attr(), "epsilon", tf_epsilon) == false)
                {
                    std::cout << "epsilon attribute not present, setting to 0.0001";
                    // TensorFlow default
                    tf_epsilon = 0.0001;
                }
                BatchToNGraph(is_nhwc, ng_input);
                std::shared_ptr<ngraph::Node> ng_batch_norm =
                    std::make_shared<ngraph::op::BatchNormInference>(
                        tf_epsilon, ng_scale, ng_offset, ng_input, ng_mean, ng_variance);
                BatchToTensorflow(is_nhwc, ng_batch_norm);

                ng_batch_norm->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_batch_norm}};

                return ret;
            }

            NamedNodeVector TranslateConcatV2Op(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                ngraph::op::ParameterVector& parameters)
            {
                const int input_cnt = node.input_size();
                if (input_cnt < 3)
                {
                    std::cerr << "\"" << node.name() << "\" requires at least 3 inputs, got "
                              << input_cnt << " instead";
                    assert(false);
                }
                ngraph::NodeVector ng_args;
                for (int i = 0; i < input_cnt - 1; i++)
                {
                    auto ng_arg = GetInputNode(all_ng_nodes, node, i);
                    ng_args.push_back(ng_arg);
                }

                auto ng_concat_axis_op = GetInputNode(all_ng_nodes, node, input_cnt - 1);
                std::vector<int> tf_concat_axis_vec;
                assert(GetValueFromNGraphOp<int>(ng_concat_axis_op, &tf_concat_axis_vec) == true);

                int64 concat_axis = tf_concat_axis_vec[0];

                if (concat_axis < 0)
                {
                    concat_axis += int64(ng_args[0]->get_shape().size());
                }
                std::shared_ptr<ngraph::Node> ng_concat_op =
                    std::make_shared<ngraph::op::Concat>(ng_args, size_t(concat_axis));
                ng_concat_op->set_name(node.name());

                NamedNodeVector ret{{node.name(), ng_concat_op}};
                return ret;
            }

            NamedNodeVector TranslateSigmoidOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto exp_op = std::make_shared<ngraph::op::Exp>(
                    std::make_shared<ngraph::op::Negative>(ng_input));
                auto constant_1 = std::make_shared<ngraph::op::Constant>(
                    ng_input->get_element_type(),
                    ng_input->get_shape(),
                    std::vector<std::string>(ngraph::shape_size(ng_input->get_shape()), "1"));
                auto denominator_op = std::make_shared<ngraph::op::Add>(constant_1, exp_op);

                auto ng_sigmoid_op =
                    std::make_shared<ngraph::op::Divide>(constant_1, denominator_op);
                ng_sigmoid_op->set_name(node.name());

                NamedNodeVector ret{{node.name(), ng_sigmoid_op}};
                return ret;
            }

            NamedNodeVector TranslateSumOp(const tensorflow::NodeDef& node,
                                           const NodeMap& all_ng_nodes,
                                           ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_axes_op = GetInputNode(all_ng_nodes, node, 1);

                bool tf_keep_dims;
                if (GetNodeAttr(node.attr(), "keep_dims", tf_keep_dims) == false)
                {
                    if (GetNodeAttr(node.attr(), "keepdims", tf_keep_dims) == false)
                    {
                        tf_keep_dims = false;
                    }
                }

                std::vector<int64> sum_axes;
                assert(GetValueFromNGraphOp<int64>(ng_axes_op, &sum_axes) == true);

                ngraph::Shape input_shape = ng_input->get_shape();
                size_t input_rank = input_shape.size();

                assert(CheckAxisDimInRange(sum_axes, input_rank));

                std::vector<size_t> ng_reduction_axes_vect(sum_axes.size());
                std::transform(
                    sum_axes.begin(),
                    sum_axes.end(),
                    ng_reduction_axes_vect.begin(),
                    [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
                ngraph::AxisSet ng_reduction_axes(ng_reduction_axes_vect);

                std::shared_ptr<ngraph::Node> ng_sum_op =
                    std::make_shared<ngraph::op::Sum>(ng_input, ng_reduction_axes);
                // If keep_dims is specified we need to reshape to put back the reduced
                // axes, with length 1.
                if (tf_keep_dims)
                {
                    ngraph::Shape ng_result_shape_with_keep(input_rank);

                    for (size_t i = 0; i < input_rank; i++)
                    {
                        ng_result_shape_with_keep[i] =
                            ng_reduction_axes.count(i) == 0 ? input_shape[i] : 1;
                    }
                    ngraph::AxisVector ng_axis_order(ng_sum_op->get_shape().size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                    ng_sum_op = std::make_shared<ngraph::op::Reshape>(
                        ng_sum_op, ng_axis_order, ng_result_shape_with_keep);
                }
                ng_sum_op->set_name(node.name());

                NamedNodeVector ret{{node.name(), ng_sum_op}};
                return ret;
            }

            NamedNodeVector TranslateSplitOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             ngraph::op::ParameterVector& parameters)
            {
                auto ng_split_dim = GetInputNode(all_ng_nodes, node, 0);
                auto ng_input = GetInputNode(all_ng_nodes, node, 1);

                // num_split : The number of ways to split. Must evenly divide
                // value.shape[split_dim]
                int32 num_split;
                assert(GetNodeAttr(node.attr(), "num_split", num_split) == true);
                ngraph::Shape shape = ng_input->get_shape();
                int rank = shape.size();
                std::vector<size_t> lower;
                std::vector<size_t> upper;
                for (int i = 0; i < rank; ++i)
                {
                    lower.push_back(0);
                    upper.push_back(shape[i]);
                }
                std::vector<int> split_dim_vec;
                assert(GetValueFromNGraphOp<int>(ng_split_dim, &split_dim_vec) == true);
                int split_dim = split_dim_vec[0] + (split_dim_vec[0] < 0 ? (int64)rank : 0);
                int size = shape[split_dim] / num_split;
                int cursor = 0;

                std::vector<std::shared_ptr<ngraph::Node>> ng_split_op_list;

                for (size_t i = 0; i < num_split; ++i)
                {
                    lower[split_dim] = cursor;
                    cursor += size;
                    upper[split_dim] = cursor;
                    auto ng_split_op = std::make_shared<ngraph::op::Slice>(ng_input, lower, upper);
                    //ng_split_op->set_name(node.name());
                    ng_split_op_list.push_back(ng_split_op);
                }
                NamedNodeVector ret;
                for (int i = 0; i < ng_split_op_list.size(); i++)
                {
                    std::string node_name = node.name();
                    //if (i > 0)
                    //{
                    //    node_name.append("_").append(std::to_string(i));
                    //}
                    ret.push_back({node_name, ng_split_op_list[i]});
                }
                return ret;
            }

            NamedNodeVector TranslateSplitVOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_length_op = GetInputNode(all_ng_nodes, node, 1);
                auto ng_split_dim = GetInputNode(all_ng_nodes, node, 2);

                std::vector<int> lengths;
                assert(GetValueFromNGraphOp<int>(ng_length_op, &lengths) == true);
                ngraph::Shape shape = ng_input->get_shape();
                int rank = shape.size();
                std::vector<size_t> lower(rank, 0);
                std::vector<size_t> upper(shape);

                std::vector<int64> split_dim_vec;
                assert(GetValueFromNGraphOp<int64>(ng_split_dim, &split_dim_vec) == true);
                // there should be at least one element specified as axis and not more than
                // one as axis is 0-D
                if (split_dim_vec.size() != 1)
                {
                    std::cerr << "split_dim_tensor must have exactly one element.";
                    assert(false);
                }
                assert(CheckAxisDimInRange(split_dim_vec, rank));

                int split_dim = split_dim_vec[0] + (split_dim_vec[0] < 0 ? (int64)rank : 0);

                // length: Length of size_splits
                int length = 0;
                int idx = -1;
                // Find out the total length of the splits and locate -1 's index, if any
                bool has_one_neg = false;
                for (int i = 0; i < lengths.size(); ++i)
                {
                    if (lengths[i] != -1)
                    {
                        length += lengths[i];
                    }
                    else
                    {
                        if (has_one_neg)
                        {
                            std::cerr << "size_splits can only have one -1";
                            assert(false);
                        }
                        else
                        {
                            idx = i;
                            has_one_neg = true;
                        }
                    }
                }

                // Size splits must sum to the dimension of value along split_dim
                if (idx > 0)
                {
                    lengths[idx] = shape[split_dim] - length;
                }

                if ((!has_one_neg && length != shape[split_dim]) ||
                    (has_one_neg && lengths[idx] < 0))
                {
                    std::cerr << "The length of size_splits must sum to the value of the dimension "
                                 "along split_dim";
                    assert(false);
                }
                int cursor = 0;
                std::vector<std::shared_ptr<ngraph::Node>> ng_split_op_list;
                if (lengths.size() != 1)
                {
                    for (int i = 0; i < lengths.size(); ++i)
                    {
                        lower[split_dim] = cursor;
                        cursor += lengths[i];
                        upper[split_dim] = cursor;
                        auto ng_split_op =
                            std::make_shared<ngraph::op::Slice>(ng_input, lower, upper);
                        //ng_split_op->set_name(node.name());
                        ng_split_op_list.push_back(ng_split_op);
                    }
                }
                else
                {
                    ng_split_op_list.push_back(ng_input);
                }

                NamedNodeVector ret;
                for (int i = 0; i < ng_split_op_list.size(); i++)
                {
                    std::string node_name = node.name();
                    //if (i > 0)
                    //{
                    //    node_name.append("_").append(std::to_string(i));
                    //}

                    ret.push_back({node_name, ng_split_op_list[i]});
                }
                return ret;
            }

            NamedNodeVector TranslateMeanOp(const tensorflow::NodeDef& node,
                                            const NodeMap& all_ng_nodes,
                                            ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_axes_op = GetInputNode(all_ng_nodes, node, 1);

                bool tf_keep_dims;
                if (GetNodeAttr(node.attr(), "keep_dims", tf_keep_dims) == false)
                {
                    if (GetNodeAttr(node.attr(), "keepdims", tf_keep_dims) == false)
                    {
                        tf_keep_dims = false;
                    }
                }

                ngraph::Shape shape = ng_input->get_shape();
                int rank = shape.size();

                std::vector<int64> mean_axes;
                assert(GetValueFromNGraphOp<int64>(ng_axes_op, &mean_axes) == true);

                assert(CheckAxisDimInRange(mean_axes, rank));

                std::vector<size_t> ng_reduction_axes_vect(mean_axes.size());
                std::transform(mean_axes.begin(),
                               mean_axes.end(),
                               ng_reduction_axes_vect.begin(),
                               [rank](int idx) { return idx + (idx < 0 ? (int)rank : 0); });
                ngraph::AxisSet ng_reduction_axes(ng_reduction_axes_vect);

                std::shared_ptr<ngraph::Node> ng_mean =
                    ngraph::builder::mean(ng_input, ng_reduction_axes);

                // If keep_dims is specified we need to reshape to put back the reduced
                // axes, with length 1.
                if (tf_keep_dims)
                {
                    ngraph::Shape ng_result_shape_with_keep(rank);
                    for (size_t i = 0; i < rank; i++)
                    {
                        ng_result_shape_with_keep[i] =
                            ng_reduction_axes.count(i) == 0 ? shape[i] : 1;
                    }

                    ngraph::AxisVector ng_axis_order(ng_mean->get_shape().size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                    ng_mean = std::make_shared<ngraph::op::Reshape>(
                        ng_mean, ng_axis_order, ng_result_shape_with_keep);
                }
                ng_mean->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_mean}};
                return ret;
            }

            NamedNodeVector TranslateSliceOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_begin = GetInputNode(all_ng_nodes, node, 1);
                auto ng_size = GetInputNode(all_ng_nodes, node, 2);

                std::vector<int64> lower_vec;
                std::vector<int64> size_vec;
                assert(GetValueFromNGraphOp<int64>(ng_begin, &lower_vec) == true);
                assert(GetValueFromNGraphOp<int64>(ng_size, &size_vec) == true);

                if (lower_vec.size() != size_vec.size())
                {
                    std::cerr << "Cannot translate sliceop: Size of lower = " << lower_vec.size()
                              << ", size of size_vec = " << size_vec.size()
                              << ". Expected them to match.";
                    assert(false);
                }

                std::vector<int> upper_vec(lower_vec.size());
                const auto ng_input_shape = ng_input->get_shape();
                std::stringstream err_stream;
                std::string err_msg;
                for (size_t i = 0; i < size_vec.size(); i++)
                {
                    if (size_vec[i] != -1)
                    {
                        upper_vec[i] = lower_vec[i] + size_vec[i];
                    }
                    else
                    {
                        // support -1 for size_vec, to the end of the tensor
                        upper_vec[i] = ng_input_shape[i];
                    }

                    // check for this condition: 0 <= begin[i] <= begin[i] + size[i] <= Di
                    if (0 > lower_vec[i])
                    {
                        err_stream << "lower < 0: " << lower_vec[i]
                                   << ". It should have been positive.\n";
                    }
                    if (lower_vec[i] > upper_vec[i])
                    {
                        err_stream << "upper < lower: upper = " << upper_vec[i]
                                   << ", lower = " << lower_vec[i] << "\n";
                    }
                    if (upper_vec[i] > ng_input_shape[i])
                    {
                        err_stream << "dim < upper: dim = " << ng_input_shape[i]
                                   << ", upper = " << upper_vec[i] << "\n";
                    }

                    err_msg = err_stream.str();
                    if (!err_msg.empty())
                    {
                        std::cerr << "Cannot translate sliceop at position " << i << " of "
                                  << size_vec.size() << ". The reasons are:\n"
                                  << err_msg;
                        assert(false);
                    }
                }

                std::vector<size_t> l(lower_vec.begin(), lower_vec.end());
                std::vector<size_t> u(upper_vec.begin(), upper_vec.end());
                auto ng_slice = std::make_shared<ngraph::op::Slice>(ng_input, l, u);

                ng_slice->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_slice}};
                return ret;
            }

            NamedNodeVector TranslateTransposeOp(const tensorflow::NodeDef& node,
                                                 const NodeMap& all_ng_nodes,
                                                 ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_permutation_op = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int64> permutation;
                assert(GetValueFromNGraphOp<int64>(ng_permutation_op, &permutation) == true);

                ngraph::AxisVector ng_axis_order;
                ng_axis_order.reserve(permutation.size());

                for (auto i : permutation)
                {
                    ng_axis_order.push_back(i);
                }

                ngraph::op::OpConfig::any myConfig;
                myConfig["axes_order"] = ng_axis_order;

                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(),
                    node.op(),
                    std::vector<std::shared_ptr<Node>>({ng_input}),
                    myConfig);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateTransposeToReshapeOp(const tensorflow::NodeDef& node,
                                                          const NodeMap& all_ng_nodes,
                                                          ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_permutation_op = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int64> permutation;
                assert(GetValueFromNGraphOp<int64>(ng_permutation_op, &permutation) == true);

                // Check to make sure that the permutation requested for transpose
                // is valid for example:
                // - it should not have duplicates,
                // - it should have all the dimensions.

                auto ng_input_rank = ng_input->get_shape().size();
                vector<bool> count(ng_input_rank, false);
                for (auto p : permutation)
                {
                    if (0 <= p && p < ng_input_rank)
                    {
                        count[p] = true;
                    }
                }
                for (int i = 0; i < ng_input_rank; i++)
                {
                    if (!count[i])
                    {
                        std::cerr << i << " is missing from {" << join(permutation) << "}.";
                        assert(false);
                    }
                }

                ngraph::AxisVector ng_axis_order;
                ng_axis_order.reserve(permutation.size());

                for (auto i : permutation)
                {
                    ng_axis_order.push_back(i);
                }

                auto ng_node = ngraph::builder::numpy_transpose(ng_input, ng_axis_order);
                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateOneHotOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              ngraph::op::ParameterVector& parameters)
            {
                auto ng_features = GetInputNode(all_ng_nodes, node, 0);
                auto ng_depth_op = GetInputNode(all_ng_nodes, node, 1);
                auto ng_on = GetInputNode(all_ng_nodes, node, 2);
                auto ng_off = GetInputNode(all_ng_nodes, node, 3);

                auto ng_features_shape = ng_features->get_shape();
                auto ng_features_rank = ng_features_shape.size();

                std::vector<int> depth;
                assert(GetValueFromNGraphOp<int>(ng_depth_op, &depth) == true);
                if (depth.size() != 1)
                {
                    std::cerr << "OneHot Op: depth of one hot dimension must be scalar "
                              << depth.size();
                    assert(false);
                }
                std::vector<float> on_value;
                assert(GetValueFromNGraphOp<float>(ng_on, &on_value) == true);
                if (on_value.size() != 1)
                {
                    std::cerr << "OneHot Op: on value of one hot dimension must be scalar "
                              << on_value.size();
                    assert(false);
                }
                std::vector<float> off_value;
                assert(GetValueFromNGraphOp<float>(ng_off, &off_value) == true);
                if (off_value.size() != 1)
                {
                    std::cerr << "OneHot Op: off value of one hot dimension must be scalar "
                              << off_value.size();
                    assert(false);
                }

                int one_hot_axis;
                assert(GetNodeAttr(node.attr(), "axis", one_hot_axis) == true);
                tensorflow::DataType dtype;
                assert(GetNodeAttr(node.attr(), "T", dtype) == true);
                ngraph::element::Type ng_et;
                assert(TFDataTypeToNGraphElementType(dtype, &ng_et) == true);

                assert(ng_et == ngraph::element::f32);

                ngraph::op::OpConfig::any myConfig;
                myConfig["axis"] = one_hot_axis;
                myConfig["depth"] = depth[0];
                myConfig["off_value"] = off_value[0];
                myConfig["on_value"] = on_value[0];
                myConfig["T"] = ng_et.c_type_string();

                //ng_features->set_output_type(0, ng_et, ng_features->get_shape());

                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(),
                    node.op(),
                    std::vector<std::shared_ptr<Node>>({ng_features}),
                    myConfig);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateStopGradientOp(const tensorflow::NodeDef& node,
                                                    const NodeMap& all_ng_nodes,
                                                    ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);

                ngraph::op::OpConfig::any myConfig;

                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(),
                    node.op(),
                    std::vector<std::shared_ptr<Node>>({ng_input}),
                    myConfig);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateGatherV2Op(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_input_coords = GetInputNode(all_ng_nodes, node, 1);
                auto ng_axis_op = GetInputNode(all_ng_nodes, node, 2);

                std::vector<int64> tf_axis;
                assert(GetValueFromNGraphOp<int64>(ng_axis_op, &tf_axis) == true);
                if (tf_axis.size() > 1)
                {
                    std::cerr << "Found axis in GatherV2 op (" << node.name()
                              << ") translation to be non scalar, of size " << tf_axis.size();
                    assert(false);
                }

                ngraph::op::OpConfig::any myConfig;
                myConfig["axis"] = tf_axis[0];

                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(),
                    node.op(),
                    std::vector<std::shared_ptr<Node>>({ng_input, ng_input_coords}),
                    myConfig);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateAddNOp(const tensorflow::NodeDef& node,
                                            const NodeMap& all_ng_nodes,
                                            ngraph::op::ParameterVector& parameters)
            {
                // Use this to get all the inputs of current node.
                // Use GetInputNode(..., ..., id) to get the input identified by
                // id.
                auto inputs = GetAllInputNode(all_ng_nodes, node);

                ngraph::op::OpConfig::any myConfig;

                // Since Ngraph doesn't have AddN, so we use GenericOp to
                // represent the AddN.
                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(), // Node name, looks like "tf_model/add_n";
                    node.op(),   // Operator name, looks like "AddN";
                    inputs,      // The inputs of nodes;
                    myConfig);   // The configuration we generated above;

                ng_node->set_name(node.name()); // Set the node name;
                // Return the node vecoter, this is one tf-node to one nnfusion-node case,
                // if your code converts one tf-node into several nnfusion-nodes, you can
                // refer BiasAdd, which is converted to Broadcast and Add;
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslatePackOp(const tensorflow::NodeDef& node,
                                            const NodeMap& all_ng_nodes,
                                            ngraph::op::ParameterVector& parameters)
            {
                const int input_cnt = node.input_size();
                if (input_cnt < 1)
                {
                    std::cerr << "\"" << node.name() << "\" requires at least 1 inputs, got "
                              << input_cnt << " instead";
                    assert(false);
                }
                int pack_axis = 0;
                assert(GetNodeAttr(node.attr(), "axis", pack_axis) == true);

                ngraph::NodeVector ng_args;
                for (int i = 0; i < input_cnt; i++)
                {
                    auto ng_arg = GetInputNode(all_ng_nodes, node, i);
                    ng_args.push_back(ng_arg);
                }

                if (pack_axis < 0)
                {
                    pack_axis += int64(ng_args[0]->get_shape().size() + 1);
                }

                if (true)
                {
                    // option1, covert pack to combination of expand_dim and concat
                    auto& input_shape = ng_args[0]->get_shape();
                    auto input_shape_size = input_shape.size();

                    // expand_dim/reshape
                    auto new_dim_shape = input_shape;
                    new_dim_shape.insert(new_dim_shape.begin() + size_t(pack_axis), 1);
                    std::vector<size_t> shape_dimensions(input_shape_size);
                    std::iota(shape_dimensions.begin(), shape_dimensions.end(), 0);

                    ngraph::NodeVector reshaped_ng_args;

                    for (int i = 0; i < ng_args.size(); i++)
                    {
                        auto ng_arg = ng_args[i];
                        auto ng_node = std::make_shared<ngraph::op::Reshape>(
                            ng_arg, shape_dimensions, new_dim_shape);
                        ng_node->set_name(node.name() + "_reshape_" + std::to_string(i));
                        reshaped_ng_args.push_back(ng_node);
                    }

                    // concat
                    std::shared_ptr<ngraph::Node> ng_concat_op =
                        std::make_shared<ngraph::op::Concat>(reshaped_ng_args, size_t(pack_axis));
                    ng_concat_op->set_name(node.name());

                    NamedNodeVector ret{{node.name(), ng_concat_op}};
                    return ret;
                }
                else
                {
                    // TODO: option2, implement pack kernel
                    std::cerr << "Pack kernel not implemented yet";
                    assert(false);

                    ngraph::op::OpConfig::any myConfig;
                    myConfig["axis"] = pack_axis;

                    auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                        node.name(), node.op(), ng_args, myConfig);

                    ng_node->set_name(node.name());

                    NamedNodeVector ret{{node.name(), ng_node}};
                    return ret;
                }
            }

            NamedNodeVector TranslateAllOp(const tensorflow::NodeDef& node,
                                           const NodeMap& all_ng_nodes,
                                           ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_axis_op = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int> tf_axis;
                assert(GetValueFromNGraphOp<int>(ng_axis_op, &tf_axis) == true);
                if (tf_axis.size() > 1)
                {
                    std::cerr << "Found axis in All op (" << node.name()
                              << ") translation to be non scalar, of size " << tf_axis.size();
                    assert(false);
                }

                bool keep_dims = false;
                assert(GetNodeAttr(node.attr(), "keep_dims", keep_dims) == true);

                ngraph::op::OpConfig::any myConfig;
                if (tf_axis.size() > 0)
                {
                    myConfig["axis"] = tf_axis[0];
                }
                myConfig["keep_dims"] = keep_dims;

                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(),
                    node.op(),
                    std::vector<std::shared_ptr<Node>>({ng_input}),
                    myConfig);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateSqueezeOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                size_t input_dims = ng_input->get_shape().size();

                std::vector<int32> tf_axis;
                assert(GetNodeAttr(node.attr(), "squeeze_dims", tf_axis) == true);

                // If input dimension is negative, make it positive
                for (int i = 0; i < tf_axis.size(); i++)
                {
                    tf_axis[i] = tf_axis[i] < 0 ? (int32)(input_dims) + tf_axis[i] : tf_axis[i];
                }

                std::set<int> axis_set(tf_axis.begin(), tf_axis.end());
                ngraph::Shape input_shape = ng_input->get_shape();
                std::vector<int> dims;

                if (axis_set.size() == 0)
                {
                    for (size_t i = 0; i < input_dims; i++)
                    {
                        if (input_shape[i] > 1)
                        {
                            dims.push_back(input_shape[i]);
                        }
                    }
                }
                else
                {
                    for (size_t i = 0; i < input_dims; i++)
                    {
                        bool skip = false;
                        if (axis_set.find(i) != axis_set.end())
                        {
                            if (input_shape[i] == 1)
                            {
                                skip = true;
                            }
                            else
                            {
                                std::cerr << "Tried to explicitly squeeze dimension " << i
                                          << " but dimension was not 1: " << input_shape[i];
                                assert(false);
                            }
                        }
                        if (!skip)
                        {
                            dims.push_back(input_shape[i]);
                        }
                    }
                }

                ngraph::Shape output_shape(dims.size());
                for (size_t i = 0; i < dims.size(); ++i)
                {
                    output_shape[i] = dims[i];
                }

                ngraph::AxisVector ng_axis_order(ng_input->get_shape().size());
                std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

                auto ng_node =
                    std::make_shared<ngraph::op::Reshape>(ng_input, ng_axis_order, output_shape);
                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateExpandDimsOp(const tensorflow::NodeDef& node,
                                                  const NodeMap& all_ng_nodes,
                                                  ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_dim = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int64> dim_vec;
                assert(GetValueFromNGraphOp<int64>(ng_dim, &dim_vec) == true);

                if (dim_vec.size() != 1)
                {
                    std::cerr << "The size of argument dim is not 1 for ExpandDims";
                    assert(false);
                }

                auto& shape = ng_input->get_shape();
                auto shape_size = shape.size();
                if (dim_vec[0] < 0)
                {
                    // allow range [-rank(input) - 1, rank(input)]
                    // where -1 append new axis at the end
                    dim_vec[0] = shape_size + dim_vec[0] + 1;
                }

                auto out_shape = shape;
                out_shape.insert(out_shape.begin() + size_t(dim_vec[0]), 1);
                std::vector<size_t> shape_dimensions(shape.size());
                std::iota(shape_dimensions.begin(), shape_dimensions.end(), 0);

                auto ng_node =
                    std::make_shared<ngraph::op::Reshape>(ng_input, shape_dimensions, out_shape);
                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateSquaredDifferenceOp(const tensorflow::NodeDef& node,
                                                         const NodeMap& all_ng_nodes,
                                                         ngraph::op::ParameterVector& parameters)
            {
                auto ng_lhs = GetInputNode(all_ng_nodes, node, 0);
                auto ng_rhs = GetInputNode(all_ng_nodes, node, 1);

                std::tie(ng_lhs, ng_rhs) =
                    ngraph::builder::numpy_broadcast(std::make_pair(ng_lhs, ng_rhs));

                auto ng_diff = std::make_shared<ngraph::op::Subtract>(ng_lhs, ng_rhs);

                auto ng_node = std::make_shared<ngraph::op::Multiply>(ng_diff, ng_diff);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateRangeOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             ngraph::op::ParameterVector& parameters)
            {
                std::vector<std::shared_ptr<Node>> input_nodes;
                auto start_node = GetInputNode(all_ng_nodes, node, 0);
                input_nodes.push_back(start_node);
                auto limit_node = GetInputNode(all_ng_nodes, node, 1);
                input_nodes.push_back(limit_node);
                auto delta_node = GetInputNode(all_ng_nodes, node, 2);
                input_nodes.push_back(delta_node);

                std::vector<int64> start_vec;
                CHECK(GetValueFromNGraphOp<int64>(start_node, &start_vec) == true);
                CHECK(start_vec.size() > 0);
                std::vector<int64> limit_vec;
                CHECK(GetValueFromNGraphOp<int64>(limit_node, &limit_vec) == true);
                CHECK(limit_vec.size() > 0);
                std::vector<int64> delta_vec;
                CHECK(GetValueFromNGraphOp<int64>(delta_node, &delta_vec) == true);
                CHECK(delta_vec.size() > 0);

                ngraph::op::OpConfig::any myConfig;
                myConfig["start"] = start_vec[0];
                myConfig["limit"] = limit_vec[0];
                myConfig["delta"] = delta_vec[0];

                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(), node.op(), input_nodes, myConfig);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateRsqrtOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);

                // Create a constant tensor populated with the value -1/2.
                // (1/sqrt(x) = x^(-1/2))
                auto shape = ng_input->get_shape();
                std::vector<std::string> constant_values(ngraph::shape_size(shape), "-0.5");

                auto ng_exponent = std::make_shared<ngraph::op::Constant>(
                    ng_input->get_element_type(), shape, constant_values);

                // Raise each element of the input to the power -0.5.
                auto ng_node = std::make_shared<ngraph::op::Power>(ng_input, ng_exponent);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateRsqrtGradOp(const tensorflow::NodeDef& node,
                                                 const NodeMap& all_ng_nodes,
                                                 ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_delta = GetInputNode(all_ng_nodes, node, 1);

                //`grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
                // Create a constant tensor populated with the value 3.
                auto et = ng_input->get_element_type();
                auto shape = ng_input->get_shape();
                std::vector<std::string> constant_values(ngraph::shape_size(shape), "3");

                auto ng_exponent =
                    std::make_shared<ngraph::op::Constant>(et, shape, constant_values);

                // Raise each element of the input to the power 3.
                auto ng_pow = std::make_shared<ngraph::op::Power>(ng_input, ng_exponent);

                // Create a constant tensor populated with the value -1/2.
                std::vector<std::string> constant_diff(ngraph::shape_size(shape), "-0.5");
                auto ng_diff = std::make_shared<ngraph::op::Constant>(et, shape, constant_diff);
                auto ng_multiply = std::make_shared<ngraph::op::Multiply>(ng_pow, ng_delta);
                auto ng_node = std::make_shared<ngraph::op::Multiply>(ng_multiply, ng_diff);
                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateStridedSliceOp(const tensorflow::NodeDef& node,
                                                    const NodeMap& all_ng_nodes,
                                                    ngraph::op::ParameterVector& parameters)
            {
                // TODO: implement new_axis_mask, ellipsis_mask
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_begin_op = GetInputNode(all_ng_nodes, node, 1);
                auto ng_end_op = GetInputNode(all_ng_nodes, node, 2);
                auto ng_stride_op = GetInputNode(all_ng_nodes, node, 3);

                std::vector<int64> begin_vec;
                assert(GetValueFromNGraphOp<int64>(ng_begin_op, &begin_vec) == true);
                std::vector<int64> end_vec;
                assert(GetValueFromNGraphOp<int64>(ng_end_op, &end_vec) == true);
                std::vector<int64> stride_vec;
                assert(GetValueFromNGraphOp<int64>(ng_stride_op, &stride_vec) == true);

                int tf_shrink_axis_mask;
                assert(GetNodeAttr(node.attr(), "shrink_axis_mask", tf_shrink_axis_mask) == true);

                int tf_end_mask;
                assert(GetNodeAttr(node.attr(), "end_mask", tf_end_mask) == true);

                int tf_begin_mask;
                assert(GetNodeAttr(node.attr(), "begin_mask", tf_begin_mask) == true);

                int tf_new_axis_mask;
                assert(GetNodeAttr(node.attr(), "new_axis_mask", tf_new_axis_mask) == true);

                int tf_ellipsis_mask;
                assert(GetNodeAttr(node.attr(), "ellipsis_mask", tf_ellipsis_mask) == true);

                auto& input_shape = ng_input->get_shape();

                // Summary: Convert tf indexes (-inf, inf) to clamped_begin_idx [0, d] and
                // clamped_end_idx [-1, d], which are then converted to ngraph indexes [0, d]
                // tf->ng is done through tf_to_ng, which calls clamper, which converts
                // tf->clamped

                // Graph/function for tf->cmapled
                //           |    .......     <-- y = max_val (max_val = 5)
                //          .|   .
                //         . |  .
                //        .  | .              <-- y = x>=0 ? x : x+max_val
                //       .   |.
                // -.-.-.----.------------    <-- y = 0 (for inclusive)
                //  * *      |                <-- y = -1 (for exclusive)
                //           |
                // X axis: TF indexes. Y axis: Clamped indexes

                // clamper is a function that implements the graph above.
                // For inclusive, the graph is clamped at 0 and dim-1
                // Given dimension d, [0, d-1] are valid locations.
                // -1 represents std::rend(). d represents std::end().
                // These two are useful for representing exclusive boundaries for end-ranges
                // Example for dim = 3:
                // ranges:                 (-inf,-d)|   [-d,0)    |[0,d-1]|(d-1,inf)
                // TF index:                  -5 -4 |-3  -2 -1    | 0 1 2 | 3 4 5
                // clamped begin (inclusive):  0  0 | 0   1  2    | 0 1 2 | 3 3 3
                // clamped end (exclusive):   -1 -1 | 0   1  2    | 0 1 2 | 3 3 3
                auto clamper = [](int idx, size_t dim, bool inclusive) {
                    // if idx is in [-(d-1), d-1], then its same for both inclusive and
                    // exclusive
                    // The first 2 cases breaks down this range
                    if (idx >= 0 && idx <= (static_cast<int>(dim) - 1))
                    {
                        return idx;
                    }
                    else if (idx < 0 && idx + static_cast<int>(dim) >= 0)
                    { // careful not to do idx >= -dim
                        // (since dim is unsigned)
                        return idx + static_cast<int>(
                                         dim); // Type casting to int to enable unambiguous auto
                                               // type inference of return type
                    }
                    else if (idx > static_cast<int>(dim) - 1)
                    {
                        return static_cast<int>(dim);
                    }
                    else if (idx + static_cast<int>(dim) < 0)
                    {
                        // The next case handles the clamping (differently for inclusive and
                        // exclusive cases)

                        // careful not to do idx < -dim (since dim is unsigned)
                        return 0 - (inclusive ? 0 : 1);
                    }
                    // Default case
                    return 0;
                };

                auto tf_to_ng = [clamper](int tf_begin_idx,
                                          int tf_end_idx,
                                          int tf_stride,
                                          size_t dim,
                                          bool begin_mask,
                                          bool end_mask,
                                          bool shrink_mask) {
                    // if begin mask is present, depending on stride sign use 0 (std::begin) or
                    // dim-1 (std::rbegin)
                    // clamped_end_idx could line in [-1, d]
                    int tf_ignore_begin_if_needed =
                        begin_mask ? (tf_stride > 0 ? 0 : dim - 1) : tf_begin_idx;
                    // if end mask is present, depending on stride sign use -1 (std::rend) or
                    // dim (std::end).
                    // However note, we cannot set to -1, since it has another meaning, hence
                    // setting to -(dim+1), which would translate to -1 in clamped coordinates
                    // take care to convert dim from sixze_t to int
                    int tf_ignore_end_if_needed =
                        end_mask ? (tf_stride > 0 ? dim : (-((int)dim + 1))) : tf_end_idx;
                    // using size_t for clamped_begin_idx because: clamped_begin_idx is
                    // inclusive, so it must lie in [0, dim-1]
                    size_t clamped_begin_idx = clamper(tf_ignore_begin_if_needed, dim, true);
                    int64 clamped_end_idx = clamper(
                        shrink_mask ? clamped_begin_idx + 1 : tf_ignore_end_if_needed, dim, false);

                    // Now we have converted semantically non-monotonic and unbounded TF indexes
                    // (-inf, inf) to bounded and monotonic clamped indexes [-1, d]
                    // Now we need to convert clamped indexes [-1, d] to ngraph indexes [0, d]
                    // (taking care of reversal in case of negative strides)

                    size_t needs_reverse = 0;
                    size_t ng_begin_idx, ng_end_idx;

                    if (!shrink_mask)
                    {
                        if (clamped_begin_idx == clamped_end_idx)
                        {
                            // Empty due to matching indexes
                            ng_begin_idx = clamped_begin_idx;
                            // Type safety: clamped_begin_idx == clamped_end_idx implies,
                            // clamped_end_idx!=-1 (since clamped_begin_idx cannot be -1), hence end
                            // index assignment is type safe
                            ng_end_idx = clamped_end_idx;
                        }
                        else
                        { // In the whole of this else: clamped_begin_idx !=
                            // clamped_end_idx, so !(a < b) iff a > b and vice versa when
                            // comparing the indexes
                            // take care to use (int) typecase when comparing int and size_t
                            if (((int)clamped_begin_idx < clamped_end_idx) != (tf_stride > 0))
                            {
                                // Empty due to mismatching directions
                                ng_begin_idx = clamped_begin_idx;
                                // Type safe: since clamped_begin_idx is size_t (>0)
                                // [0:-4:1] in TF would convert to [0:-1:1] in clamped domain. hence
                                // we do not assign ng_end_idx = clamped_end_idx (which would not be
                                // type safe due to the -1)
                                ng_end_idx = clamped_begin_idx;
                                // Any assignment where ng_begin_idx = ng_end_idx = x (where 0 <= x <=
                                // d-1) would have worked for the 2 empty cases above
                            }
                            // Anything after this is non-empty. Anything before this has dealt with
                            // empty cases
                            else
                            {
                                // in this case either (clamped_begin_idx < clamped_end_idx &&
                                // tf_stride > 0) or (clamped_begin_idx > clamped_end_idx && tf_stride
                                // < 0)
                                // that is clamped_begin_idx < clamped_end_idx <==> tf_stride > 0.
                                // hence using only 1 of the clauses is enough
                                if (tf_stride > 0)
                                {
                                    ng_begin_idx = clamped_begin_idx;
                                    // Type safety: tf_stride > 0 ==> clamped_begin_idx <
                                    // clamped_end_idx. clamped_begin_idx could be 0,
                                    // which means clamped_end_idx > 0. Hence type-safe
                                    ng_end_idx = clamped_end_idx;
                                }
                                else
                                { // clamped_begin_idx > clamped_end_idx, tf_stride < 0

                                    // clamped_begin_idx is [0, d] && clamped_begin_idx >
                                    // clamped_end_idx,
                                    // which implies clamped_end_idx is [-1,d-1]
                                    // Type safety: With clamped_end_idx in [-1,d-1],
                                    // dim - 1 - clamped_end_idx is in [0, dim]. Hence type safe
                                    ng_end_idx = dim - 1 - clamped_end_idx;

                                    if (clamped_begin_idx == dim)
                                    {
                                        clamped_begin_idx = dim - 1;
                                    }
                                    // Note clamped_begin_idx != dim here.
                                    // If clamped_begin_idx==dim && clamped_end_idx==dim, then "Empty
                                    // due to matching indexes" handles it
                                    // If clamped_begin_idx==dim && clamped_end_idx<dim, then 2 cases:
                                    //   tf_stride > 0: then "Empty due to mismatching directions"
                                    //   handles it
                                    //   tf_stride < 0: Then we set it to dim-1 above
                                    // Consider the case of dim=3, where in tf notation we have:
                                    // [4:1:-1], in clampe notation, we get [3:1:-1], which really means
                                    // [2:1:-1]

                                    // Type safety: Since clamped_begin_idx is [0, d-1] here, it is type
                                    // safe
                                    ng_begin_idx = dim - 1 - clamped_begin_idx;
                                    needs_reverse = 1;
                                }
                            }
                        }
                    }
                    else
                    {
                        // cases when clamped indexes are in [0,d] and hence can be directly
                        // copied
                        // TODO: what about tf_begin=d, shrink=T, then clamped_end_idx = d, so a
                        // 0-d axis.
                        // But since shrink is on, that is reshaped and the 0-d axis is removed?
                        // Is that a valid config, as shrink_axis must get an axis with dim = 1,
                        // right?

                        ng_begin_idx = clamped_begin_idx;
                        ng_end_idx = clamped_end_idx;
                    }
                    return std::make_tuple(
                        ng_begin_idx, ng_end_idx, std::abs(tf_stride), needs_reverse);
                };

                auto extract_bit = [](int bit_mask, int bit_location) {
                    return (bit_mask & (1 << bit_location)) != 0;
                };

                auto dim_vec = ng_input->get_shape();
                auto in_rank = dim_vec.size();

                if (begin_vec.size() > in_rank)
                {
                    std::cerr << "Index out of range using input dim " << begin_vec.size()
                              << "; input has only " << in_rank << " dims";
                    assert(false);
                }

                // TODO/Note/Question: Are begin, end and stride vectors are of equal length

                // begin, end and stride vectors may not have same size as input rank, hence
                // initialize them with 0, dim and 1 respectively
                vector<size_t> ng_begin_vec(in_rank, 0), ng_stride_vec(in_rank, 1);
                vector<size_t> ng_end_vec(dim_vec);
                vector<size_t> ng_needs_reversal(in_rank, 0); // should have been a
                                                              // vector<bool>, but it is
                                                              // optimized, so tie won't
                                                              // work. Hence using size_t
                for (int dim_idx = 0; dim_idx < begin_vec.size(); dim_idx++)
                {
                    std::tie(ng_begin_vec[dim_idx],
                             ng_end_vec[dim_idx],
                             ng_stride_vec[dim_idx],
                             ng_needs_reversal[dim_idx]) =
                        tf_to_ng(begin_vec[dim_idx],
                                 end_vec[dim_idx],
                                 stride_vec[dim_idx],
                                 dim_vec[dim_idx],
                                 extract_bit(tf_begin_mask, dim_idx),
                                 extract_bit(tf_end_mask, dim_idx),
                                 extract_bit(tf_shrink_axis_mask, dim_idx));
                }

                // filter out negative stride dimensions
                vector<size_t> neg_strides;
                for (int dim_idx = 0; dim_idx < in_rank; dim_idx++)
                {
                    if (ng_needs_reversal[dim_idx])
                    {
                        neg_strides.push_back(dim_idx);
                    }
                }

                // atleast one stride was negative, in which case reverse the input
                if (neg_strides.size() > 0)
                {
                    ng_input = std::make_shared<ngraph::op::Reverse>(ng_input, neg_strides);
                }

                std::shared_ptr<ngraph::Node> ng_strided_slice =
                    std::make_shared<ngraph::op::Slice>(
                        ng_input, ng_begin_vec, ng_end_vec, ng_stride_vec);

                if (tf_shrink_axis_mask)
                {
                    int64 shrink_axis_mask = tf_shrink_axis_mask;
                    vector<size_t> output_shape;

                    // Note: do not use rank instead of ng_begin_vec.size()
                    // since ng_begin_vec.size() can be less than rank, and
                    // shrink_mask will have atmost ng_begin_vec.size() elements
                    for (int i = 0; i < ng_begin_vec.size(); i++)
                    {
                        if ((shrink_axis_mask & 1) != 1)
                        {
                            output_shape.push_back(ng_end_vec[i] - ng_begin_vec[i]);
                        }
                        else
                        {
                            // TODO: must it equal 1 or can it be 0 too?
                            if (ng_end_vec[i] - ng_begin_vec[i] > 1)
                            {
                                std::cerr
                                    << "Trying to shrink specification " << i
                                    << "where tf begin, end, strides are: " << begin_vec[i] << ":"
                                    << end_vec[i] << ":" << stride_vec[i]
                                    << ". nGraph begin, end, stride are: " << ng_begin_vec[i] << ":"
                                    << ng_end_vec[i] << ":" << ng_stride_vec[i]
                                    << ". nGraph's begin and end have difference greater than "
                                       "1";
                                assert(false);
                            }
                        }
                        shrink_axis_mask >>= 1;
                    }

                    ngraph::Shape ng_final_shape(output_shape);
                    ngraph::AxisVector ng_axis_order(input_shape.size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

                    ng_strided_slice = std::make_shared<ngraph::op::Reshape>(
                        ng_strided_slice, ng_axis_order, ng_final_shape);
                }

                // TODO: assert size in this dim was 1
                // TODO: assert new_axis_mask and tf_shrink_axis_mask are not set at the same
                // time?
                // TODO: tf_new_axis_mask can exceed rank
                // Raise each element of the input to the power -0.5.

                ng_strided_slice->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_strided_slice}};
                return ret;
            }

            NamedNodeVector TranslateStridedSliceGradOp(const tensorflow::NodeDef& node,
                                                        const NodeMap& all_ng_nodes,
                                                        ngraph::op::ParameterVector& parameters)
            {
                auto x = GetInputNode(all_ng_nodes, node, 0);
                auto begin = GetInputNode(all_ng_nodes, node, 1);
                auto end = GetInputNode(all_ng_nodes, node, 2);
                auto strides = GetInputNode(all_ng_nodes, node, 3);
                auto grad = GetInputNode(all_ng_nodes, node, 4);

                std::vector<int32> x_value;
                CHECK(GetValueFromNGraphOp<int32>(x, &x_value))
                    << "StridedSliceGradOp currently do not support dynamic output tensor shape";
                auto x_shape = x->get_shape();
                auto x_const =
                    std::make_shared<ngraph::op::Constant>(element::i32, x_shape, x_value);

                int tf_shrink_axis_mask;
                assert(GetNodeAttr(node.attr(), "shrink_axis_mask", tf_shrink_axis_mask) == true);

                int tf_end_mask;
                assert(GetNodeAttr(node.attr(), "end_mask", tf_end_mask) == true);

                int tf_begin_mask;
                assert(GetNodeAttr(node.attr(), "begin_mask", tf_begin_mask) == true);

                int tf_new_axis_mask;
                assert(GetNodeAttr(node.attr(), "new_axis_mask", tf_new_axis_mask) == true);

                int tf_ellipsis_mask;
                assert(GetNodeAttr(node.attr(), "ellipsis_mask", tf_ellipsis_mask) == true);

                ngraph::op::OpConfig::any myConfig;
                myConfig["begin_mask"] = tf_begin_mask;
                myConfig["end_mask"] = tf_end_mask;
                myConfig["ellipsis_mask"] = tf_ellipsis_mask;
                myConfig["new_axis_mask"] = tf_new_axis_mask;
                myConfig["shrink_axis_mask"] = tf_shrink_axis_mask;
                // TODO: change shape with mask

                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(), // Node name, looks like "tf_model/add_n";
                    node.op(),   // Operator name, looks like "AddN";
                    std::vector<std::shared_ptr<Node>>(
                        {x_const, begin, end, strides, grad}), // The inputs of nodes;
                    myConfig); // The configuration we generated above;

                ng_node->set_name(node.name()); // Set the node name;

                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateTileOp(const tensorflow::NodeDef& node,
                                            const NodeMap& all_ng_nodes,
                                            ngraph::op::ParameterVector& parameters)
            {
                /*
                This operation creates a new tensor by replicating input multiples times.
                The output tensor's i'th dimension has input.dims(i) * multiples[i] elements,
                and the values of input are replicated multiples[i] times along the 'i'th dimension.
                For example, tiling [a b c d] by [2] produces [a b c d a b c d].
                */
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_multiples = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int64> in_value;
                CHECK(GetValueFromNGraphOp<int64>(ng_multiples, &in_value))
                    << "TileOp currently do not support dynamic tensor shape";
                auto ng_input_shape = ng_multiples->get_shape();
                auto ng_const =
                    std::make_shared<ngraph::op::Constant>(element::i64, ng_input_shape, in_value);

                ngraph::op::OpConfig::any myConfig;
                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(),
                    node.op(),
                    std::vector<std::shared_ptr<Node>>({ng_input, ng_const}),
                    myConfig);
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateUnsortedSegmentSumOp(const tensorflow::NodeDef& node,
                                                          const NodeMap& all_ng_nodes,
                                                          ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_seg_id = GetInputNode(all_ng_nodes, node, 1);
                auto ng_seg_num = GetInputNode(all_ng_nodes, node, 2);

                std::vector<int> in_value;
                CHECK(GetValueFromNGraphOp<int>(ng_seg_num, &in_value))
                    << "We only accept the sgements number as Constant.";
                auto ng_const = std::make_shared<ngraph::op::Constant>(
                    element::i32, ng_seg_num->get_shape(), in_value);

                ngraph::op::OpConfig::any myConfig;
                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(),
                    node.op(),
                    std::vector<std::shared_ptr<Node>>({ng_input, ng_seg_id, ng_const}),
                    myConfig);
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateSoftmaxOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_input_shape = ng_input->get_shape();
                ngraph::AxisSet ng_axes_softmax;
                auto shape_size = ng_input_shape.size();
                if (shape_size < 1)
                {
                    std::cerr << "TF Softmax logits must be >=1 dimension";
                    assert(false);
                }
                auto rank = ng_input->get_shape().size();
                ng_axes_softmax.insert(rank - 1);

                auto ng_node = std::make_shared<ngraph::op::Softmax>(ng_input, ng_axes_softmax);
                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateAssertOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_node = std::make_shared<ngraph::op::Constant>(
                    ngraph::element::i32, ngraph::Shape{}, std::vector<int>{0});

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateSelectOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              ngraph::op::ParameterVector& parameters)
            {
                auto ng_input1 = GetInputNode(all_ng_nodes, node, 0);
                auto ng_input2 = GetInputNode(all_ng_nodes, node, 1);
                auto ng_input3 = GetInputNode(all_ng_nodes, node, 2);

                if (ng_input2->get_shape() != ng_input3->get_shape())
                {
                    std::cerr << "Input tensors 2 and 3 should have same shape";
                    assert(false);
                }

                auto ng_input1_shape = ng_input1->get_shape();
                auto ng_input2_shape = ng_input2->get_shape();

                auto ng_input1_rank = ng_input1->get_shape().size();
                auto ng_input2_rank = ng_input2->get_shape().size();

                if (!((ng_input1_shape == ng_input2_shape) ||
                      ((ng_input1_rank == 1) && (ng_input2_rank > ng_input1_rank) &&
                       (ng_input2_shape[0] == ng_input1_shape[0]))))
                {
                    std::cerr
                        << "Input tensor may have the same shape as condition. If condition is "
                        << "rank 1, input may have higher rank, but its first dimension must "
                        << "match the size of condition.";
                    assert(false);
                }

                int length = 0;
                shared_ptr<ngraph::Node> ng_input_new;

                // If input tensor has higher rank than condiiton, length will be > 0.
                length = ng_input2_rank - ng_input1_rank;

                if (length != 0)
                {
                    // Condition tensor will be modified to align the condition tensor
                    // shape with input tensor shape index and fill the rest of the vector
                    // with
                    // 1s
                    // Eg: condition tensor [7], input tensor [7, 3, 2, 1]
                    // After Reshape, condition tensor will be [7, 1 ,1 ,1] for auto
                    // broadcast.

                    std::vector<size_t> tmp_vector((ng_input2_rank), 1);
                    tmp_vector[0] = ng_input1_shape[0];

                    ng_input_new = std::make_shared<ngraph::op::Reshape>(
                        ng_input1, ngraph::AxisVector{0}, tmp_vector);
                }

                std::tie(ng_input1, ng_input2) = ngraph::builder::numpy_broadcast(
                    std::make_pair(length != 0 ? ng_input_new : ng_input1, ng_input2));
                std::tie(ng_input2, ng_input3) =
                    ngraph::builder::numpy_broadcast(std::make_pair(ng_input2, ng_input3));

                auto ng_node =
                    std::make_shared<ngraph::op::Select>(ng_input1, ng_input2, ng_input3);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector
                TranslateBroadcastGradientArgsOp(const tensorflow::NodeDef& node,
                                                 const NodeMap& all_ng_nodes,
                                                 ngraph::op::ParameterVector& parameters)
            {
                std::vector<BCast::Vec> shapes;
                for (int i = 0; i < 2; ++i)
                {
                    auto ng_input = GetInputNode(all_ng_nodes, node, i);
                    auto ng_input_shape = ng_input->get_shape();
                    CHECK(ng_input_shape.size() == 1) << "input" << i << "must be a vector";
                    std::vector<int64> in_value;
                    CHECK(GetValueFromNGraphOp<int64>(ng_input, &in_value));

                    BCast::Vec vec;
                    for (int64 i = 0; i < shape_size(ng_input_shape); ++i)
                    {
                        vec.push_back(in_value[i]);
                    }
                    shapes.push_back(vec);
                }

                BCast bcast(shapes[0], shapes[1]);
                CHECK(bcast.IsValid());
                // <<
                // "Incompatible shapes: [" << str_util::Join(shapes[0], ","),
                // "] vs. [", str_util::Join(shapes[1], ","), "]"));
                const BCast::Vec& out0 = bcast.grad_x_reduce_idx();
                const BCast::Vec& out1 = bcast.grad_y_reduce_idx();
                auto ng_out_node_0 = std::make_shared<ngraph::op::Constant>(
                    element::i64, ngraph::Shape({out0.size()}), out0);
                auto ng_out_node_1 = std::make_shared<ngraph::op::Constant>(
                    element::i64, ngraph::Shape({out1.size()}), out1);

                ng_out_node_0->set_name(node.name() + "x");
                ng_out_node_1->set_name(node.name() + "y");
                NamedNodeVector ret{{node.name(), ng_out_node_0}, {node.name(), ng_out_node_1}};

                return ret;
            }

            NamedNodeVector TranslateFloorModOp(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                ngraph::op::ParameterVector& parameters)
            {
                auto ng_input_1 = GetInputNode(all_ng_nodes, node, 0);
                auto ng_input_2 = GetInputNode(all_ng_nodes, node, 1);

                std::tie(ng_input_1, ng_input_2) =
                    ngraph::builder::numpy_broadcast(std::make_pair(ng_input_1, ng_input_2));

                std::shared_ptr<ngraph::Node> ng_floordiv_op = std::make_shared<ngraph::op::Floor>(
                    std::make_shared<ngraph::op::Divide>(ng_input_1, ng_input_2));

                std::shared_ptr<ngraph::Node> ng_floormod_op =
                    std::make_shared<ngraph::op::Subtract>(
                        ng_input_1,
                        std::make_shared<ngraph::op::Multiply>(ng_floordiv_op, ng_input_2));

                ng_floormod_op->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_floormod_op}};
                return ret;
            }
            NamedNodeVector TranslateDynamicStitchOp(const tensorflow::NodeDef& node,
                                                     const NodeMap& all_ng_nodes,
                                                     ngraph::op::ParameterVector& parameters)
            {
                int32 num_partitions;
                std::vector<std::shared_ptr<Node>> input_nodes;

                assert(GetNodeAttr(node.attr(), "N", num_partitions) == true);

                for (int i = 0; i < num_partitions * 2; i++)
                {
                    auto ng_input = GetInputNode(all_ng_nodes, node, i);
                    auto ng_input_shape = ng_input->get_shape();

                    if (i < num_partitions)
                    {
                        std::vector<int64> in_value;
                        CHECK(GetValueFromNGraphOp<int64>(ng_input, &in_value))
                            << "DynamicStitch currently do not support dynamic tensor shape";
                        auto ng_const = std::make_shared<ngraph::op::Constant>(
                            element::i64, ng_input_shape, in_value);
                        input_nodes.push_back(ng_const);
                    }
                    else
                    {
                        input_nodes.push_back(ng_input);
                    }
                }

                ngraph::op::OpConfig::any myConfig;
                myConfig["N"] = num_partitions;

                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(), node.op(), input_nodes, myConfig);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            // Computes the gradient for tanh of 'x' w.r.t its input
            // grad = dy * (1 - y * y)
            // where y = tanh(x) and dy is the corresponding input gradient
            NamedNodeVector TranslateTanhGradOp(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);
                auto ng_delta = GetInputNode(all_ng_nodes, node, 1);

                auto et = ng_input->get_element_type();
                auto input_shape = ng_input->get_shape();

                auto ng_sq = std::make_shared<ngraph::op::Multiply>(ng_input, ng_input);

                std::vector<std::string> const_values(ngraph::shape_size(input_shape), "1");

                auto ng_const =
                    std::make_shared<ngraph::op::Constant>(et, input_shape, const_values);

                auto ng_sub = std::make_shared<ngraph::op::Subtract>(ng_const, ng_sq);
                auto ng_node = std::make_shared<ngraph::op::Multiply>(ng_delta, ng_sub);
                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            NamedNodeVector TranslateFloorDivOp(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                ngraph::op::ParameterVector& parameters)
            {
                auto ng_lhs = GetInputNode(all_ng_nodes, node, 0);
                auto ng_rhs = GetInputNode(all_ng_nodes, node, 1);

                std::tie(ng_lhs, ng_rhs) =
                    ngraph::builder::numpy_broadcast(std::make_pair(ng_lhs, ng_rhs));

                auto ng_div = std::make_shared<ngraph::op::Divide>(ng_lhs, ng_rhs);

                auto ng_node = std::make_shared<ngraph::op::Floor>(ng_div);

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            const static std::map<const std::string, ConvertFunc> TRANSLATE_OP_MAP{
                {"Abs", TranslateUnaryOp<ngraph::op::Abs>},
                {"Add", TranslateBinaryOp<ngraph::op::Add>},
                {"AddN", TranslateAddNOp},
                {"All", TranslateAllOp},
                {"Assert", TranslateAssertOp},
                {"AvgPool", TranslateAvgPoolOp},
                {"BatchMatMul", TranslateBatchMatMulOp},
                {"BatchMatMulV2", TranslateBatchMatMulOp},
                {"BiasAdd", TranslateBiasAddOp},
                {"BroadcastGradientArgs", TranslateBroadcastGradientArgsOp},
                {"BiasAddGrad", TranslateBiasAddGradOp},
                {"Cast", TranslateCastOp},
                {"Const", TranslateConstOp},
                {"Conv2D", TranslateConv2DOp},
                {"ConcatV2", TranslateConcatV2Op},
                {"DivNoNan", TranslateBinaryOp<ngraph::op::DivNoNan>},
                {"DynamicStitch", TranslateDynamicStitchOp},
                {"Equal", TranslateBinaryOp<ngraph::op::Equal>},
                {"Exp", TranslateUnaryOp<ngraph::op::Exp>},
                {"ExpandDims", TranslateExpandDimsOp},
                {"Fill", TranslateFillOp},
                {"FloorMod", TranslateFloorModOp},
                {"FloorDiv", TranslateFloorDivOp},
                {"FusedBatchNorm", TranslateFusedBatchNormOp},
                {"FusedBatchNormV2", TranslateFusedBatchNormOp},
                {"GatherV2", TranslateGatherV2Op},
                {"Identity", TranslateIdentityOp},
                {"PreventGradient", TranslateIdentityOp},
                {"StopGradient", TranslateIdentityOp},
                {"NoOp", TranslateNoOp},
                {"Identity", TranslateIdentityOp},
                {"InvertPermutation", TranslateInvertPermutationOp},
                {"MatMul", TranslateMatMulOp},
                {"LessEqual", TranslateBinaryOp<ngraph::op::LessEq>},
                {"Maximum", TranslateBinaryOp<ngraph::op::Maximum>},
                {"MaxPool", TranslateMaxPoolOp},
                {"Mean", TranslateMeanOp},
                {"Mul", TranslateBinaryOp<ngraph::op::Multiply>},
                {"Multiply", TranslateBinaryOp<ngraph::op::Multiply>},
                {"Neg", TranslateUnaryOp<ngraph::op::Negative>},
                {"OneHot", TranslateOneHotOp},
                {"Pack", TranslatePackOp},
                {"Pad", TranslatePadOp},
                {"PadV2", TranslatePadV2Op},
                {"Placeholder", TranslateInputOp<ngraph::op::Parameter>},
                {"Pow", TranslateBinaryOp<ngraph::op::Power>},
                {"Range", TranslateRangeOp},
                {"Relu", TranslateUnaryOp<ngraph::op::Relu>},
                {"ReluGrad", TranslateReluGradOp},
                {"Reshape", TranslateReshapeOp},
                {"Rsqrt", TranslateRsqrtOp},
                {"RsqrtGrad", TranslateRsqrtGradOp},
                {"RealDiv", TranslateBinaryOp<ngraph::op::Divide>},
                {"Select", TranslateSelectOp},
                {"Sigmoid", TranslateSigmoidOp},
                {"Slice", TranslateSliceOp},
                {"Softmax", TranslateSoftmaxOp},
                {"Split", TranslateSplitOp},
                {"SplitV", TranslateSplitVOp},
                {"SquaredDifference", TranslateSquaredDifferenceOp},
                {"Squeeze", TranslateSqueezeOp},
                {"StridedSlice", TranslateStridedSliceOp},
                {"SparseSoftmaxCrossEntropyWithLogits",
                 TranslateSparseSoftmaxCrossEntropyWithLogitsOp},
                {"Sub", TranslateBinaryOp<ngraph::op::Subtract>},
                {"Sum", TranslateSumOp},
                {"Tanh", TranslateUnaryOp<ngraph::op::Tanh>},
                {"TanhGrad", TranslateTanhGradOp},
                {"Tile", TranslateTileOp},
                {"UnsortedSegmentSum", TranslateUnsortedSegmentSumOp},
                {"Transpose", TranslateTransposeToReshapeOp}};

            struct InputInfo
            {
                explicit InputInfo(const std::string& node_name,
                                   std::shared_ptr<nnfusion::graph::GNode> n,
                                   int i)
                    : name(node_name)
                    , node(n)
                    , index(i)
                {
                }
                std::string name;
                std::shared_ptr<nnfusion::graph::GNode> node;
                int index;
            };

            GraphConvert::GraphConvert(const tensorflow::GraphDef& proto)
                : tf_graph_proto{&proto}
            {
                LOG(INFO) << "Converting Tensorflow Graph" << std::endl;

                m_ngraph = std::make_shared<nnfusion::graph::Graph>();
                std::map<std::string, std::vector<std::shared_ptr<nnfusion::graph::GNode>>>
                    gnode_map;

                generate_topology();

                std::vector<InputInfo> inputs;
                while (!tf_topology_.empty())
                {
                    uint32_t node_idx = tf_topology_.front();
                    tf_topology_.pop();
                    inputs.clear();
                    const auto& node_proto = proto.node(node_idx);
                    bool in_control_dependence = false;
                    for (auto& input : node_proto.input())
                    {
                        TensorId input_tensor(ParseTensorName(input));
                        int src_index = input_tensor.second;

                        std::shared_ptr<nnfusion::graph::GNode> src_node;

                        auto iter = gnode_map.find(input_tensor.first);
                        if (iter == gnode_map.end())
                        {
                            std::cerr << "Node " << node_proto.name()
                                      << " has Un-Converted input node: " << input_tensor.first;
                            assert(false);
                        }
                        if (src_index == nnfusion::graph::Graph::kControlSlot)
                        {
                            in_control_dependence = true;
                            if (iter->second.size() > 0)
                            {
                                src_node = iter->second.at(0);
                                inputs.emplace_back(input_tensor.first, src_node, -1);
                            }
                        }
                        else
                        {
                            if (in_control_dependence)
                            {
                                std::cerr << "Control dependencies must come after regular "
                                             "dependencies.";
                                assert(false);
                            }
                            src_node = iter->second.at(src_index);
                            inputs.emplace_back(input_tensor.first, src_node, 0);
                        }
                    }

                    auto ng_nodes = convert_node(node_proto);
                    gnode_map[node_proto.name()] = {};
                    m_ng_node[node_proto.name()] = {};
                    for (auto& node : ng_nodes)
                    {
                        m_ng_node[node.first].push_back(node.second);
                        std::shared_ptr<nnfusion::graph::GNode> gnode = nullptr;
                        if (node2gnode_map.find(node.second) == node2gnode_map.end())
                        {
                            gnode = m_ngraph->add_node(node.second);
                            node2gnode_map[node.second] = gnode;

                            std::queue<std::shared_ptr<ngraph::Node>> process_queue;
                            process_queue.push(node.second);
                            while (!process_queue.empty())
                            {
                                auto process_node = process_queue.front();
                                process_queue.pop();
                                std::shared_ptr<nnfusion::graph::GNode> process_gnode = nullptr;
                                if (node2gnode_map.find(process_node) == node2gnode_map.end())
                                {
                                    process_gnode = m_ngraph->add_node(process_node);
                                    node2gnode_map[process_node] = process_gnode;
                                }
                                else
                                {
                                    process_gnode = node2gnode_map[process_node];
                                }
                                for (auto& process_input : process_node->get_inputs())
                                {
                                    auto process_input_node = process_input.get_output().get_node();

                                    std::shared_ptr<nnfusion::graph::GNode> process_input_gnode =
                                        nullptr;
                                    if (node2gnode_map.find(process_input_node) ==
                                        node2gnode_map.end())
                                    {
                                        process_input_gnode =
                                            m_ngraph->add_node(process_input_node);
                                        node2gnode_map[process_input_node] = process_input_gnode;
                                    }
                                    else
                                    {
                                        process_input_gnode = node2gnode_map[process_input_node];
                                    }

                                    if (!m_ngraph->find_edge(process_input_gnode,
                                                             process_input.get_output().get_index(),
                                                             process_gnode,
                                                             process_input.get_index()))
                                    {
                                        m_ngraph->add_edge(process_input_gnode,
                                                           process_input.get_output().get_index(),
                                                           process_gnode,
                                                           process_input.get_index());
                                        process_queue.push(process_input_node);
                                    }
                                }
                            }
                        }
                        else
                        {
                            gnode = node2gnode_map[node.second];
                            if (gnode->get_name() != node.first)
                            {
                                if ((*gnode)["Alias"].is_valid())
                                {
                                    std::cerr << "node " << gnode->get_name()
                                              << " has more than one alias.";
                                    assert(false);
                                }
                                (*gnode)["Alias"] = node.first;
                            }
                        }
                        gnode_map[node.first].push_back(gnode);

                        if (tf_output_name_.find(node_proto.name()) != tf_output_name_.end())
                        {
                            m_outputs.emplace_back(node.second);
                            m_graph_outputs.emplace_back(gnode);
                        }

                        for (size_t input_idx = 0; input_idx < inputs.size(); input_idx++)
                        {
                            if (inputs[input_idx].node == nullptr)
                            {
                                // todo: back edge
                                std::cerr << "Back edge is not supported now.";
                                assert(false);
                            }
                            else if (inputs[input_idx].index ==
                                     nnfusion::graph::Graph::kControlSlot)
                            {
                                m_ngraph->add_control_edge(inputs[input_idx].node, gnode);
                                node.second->add_control_dependency(
                                    inputs[input_idx].node->get_op_ptr());
                            }
                            else
                            {
                                // normal edge, do nothing
                            }
                        }
                    }

                    for (size_t i = 0; i < tf_node_outputs_[node_idx].size(); ++i)
                    {
                        const int output = tf_node_outputs_[node_idx][i];
                        tf_pending_counts_[output]--;
                        if (tf_pending_counts_[output] == 0)
                        {
                            tf_topology_.push(output);
                        }
                    }
                }

                m_ngraph->set_outputs(m_graph_outputs);
                LOG(INFO) << "convert graph done" << endl;
            }

            void GraphConvert::generate_topology()
            {
                const size_t num_nodes = tf_graph_proto->node_size();
                std::unordered_map<std::string, uint32_t> tensorflow_name2nodeIdx_map;
                for (size_t n = 0; n < num_nodes; ++n)
                {
                    tensorflow_name2nodeIdx_map[tf_graph_proto->node(n).name()] = n;
                }

                tf_pending_counts_.reserve(num_nodes);
                tf_node_outputs_.resize(num_nodes);
                for (size_t n = 0; n < num_nodes; ++n)
                {
                    const auto& node_proto = tf_graph_proto->node(n);
                    int pending_count = node_proto.input_size();
                    for (size_t i = 0; i < node_proto.input_size(); ++i)
                    {
                        std::string input_name = node_proto.input(i);
                        TensorId input_tensor(ParseTensorName(input_name));

                        auto iter = tensorflow_name2nodeIdx_map.find(input_tensor.first);
                        if (iter == tensorflow_name2nodeIdx_map.end())
                        {
                            std::cerr << "Node " << node_proto.name()
                                      << " has Unknown input node: " << input_name;
                            assert(false);
                        }
                        tf_node_outputs_[iter->second].push_back(n);
                    }
                    if (pending_count == 0)
                    {
                        tf_topology_.push(n);
                    }
                    tf_pending_counts_.push_back(pending_count);
                }

                for (size_t n = 0; n < num_nodes; ++n)
                {
                    if (tf_node_outputs_[n].size() == 0)
                    {
                        tf_output_name_.insert(tf_graph_proto->node(n).name());
                    }
                }
            }

            NamedNodeVector GraphConvert::convert_node(const tensorflow::NodeDef& node)
            {
                //LOG(INFO) << ">> ++ Managing TF_IMPORT node " << node.name();
                NamedNodeVector ret;
                auto func = TRANSLATE_OP_MAP.find(node.op());
                if (func != TRANSLATE_OP_MAP.end())
                {
                    ret = func->second(node, m_ng_node, m_parameters);
                }
                else
                {
                    // std::cerr << "Unsupport operator: " << node.op() << std::endl;
                    // return NamedNodeVector{};
                    ret = TranslateGenericNoAttrOp(node, m_ng_node, m_parameters);
                }
                //LOG(INFO) << ">> -- Managing TF_IMPORT node " << node.name();
                return std::move(ret);
            }

            std::vector<std::shared_ptr<ngraph::Function>> GraphConvert::get_funcs()
            {
                std::vector<std::shared_ptr<Function>> output_functions;
                for (const auto& output : m_outputs)
                {
                    output_functions.emplace_back(
                        std::make_shared<ngraph::Function>(output, m_parameters));
                }
                if (output_functions.size() > 1)
                    LOG(WARNING) << "Please note that NNFusion current only support single"
                                 << " output graph. If your graph has more than one outputs, we "
                                 << "ONLY generate the source code for the first one.";
                return output_functions;
            }
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace nnfusion
//----------------------------------------------------------------------------------------------
