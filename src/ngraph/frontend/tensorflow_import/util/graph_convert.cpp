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

#include "nnfusion/core/ops/generic_op.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            NamedNodeVector TranslateIdentityOp(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                ngraph::op::ParameterVector& parameters)
            {
                auto input_node = GetInputNode(all_ng_nodes, node, 0);
                NamedNodeVector ret{{node.name(), input_node}};
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
                if (transpose_a)
                {
                    ng_lhs = ngraph::builder::numpy_transpose(ng_lhs, ngraph::AxisVector{1, 0});
                }
                if (transpose_b)
                {
                    ng_rhs = ngraph::builder::numpy_transpose(ng_rhs, ngraph::AxisVector{1, 0});
                }
                
                auto ng_node = std::make_shared<ngraph::op::Dot>(ng_lhs, ng_rhs);
                ng_node->set_name(node.name());
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

                if (adj_x)
                {
                    ng_lhs = ngraph::builder::numpy_transpose(ng_lhs, ng_axis_order);
                }
                if (adj_y)
                {
                    ng_rhs = ngraph::builder::numpy_transpose(ng_rhs, ng_axis_order);
                }
                
                ngraph::op::OpConfig::any myConfig;
                myConfig["adj_x"]["b"] = false;
                myConfig["adj_y"]["b"] = false;

                auto ng_node = std::make_shared<ngraph::op::GenericOp>(
                    node.name(),
                    node.op(),
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
                    std::cerr
                        << "Constant node for paddings does not have an even number of elements";
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
                    std::cerr
                        << "Constant node for paddings does not have an even number of elements";
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
                              << ") translation to be non scalar, of size "
                              << tf_axis.size();
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
                              << ") translation to be non scalar, of size "
                              << tf_axis.size();
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
                                    << " but dimension was not 1: "
                                    << input_shape[i];
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

                auto ng_node = std::make_shared<ngraph::op::Reshape>(ng_input, ng_axis_order, output_shape);
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

                auto ng_node = std::make_shared<ngraph::op::Reshape>(ng_input, shape_dimensions, out_shape);
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
            
            NamedNodeVector TranslateRsqrtOp(const tensorflow::NodeDef& node,
                                    const NodeMap& all_ng_nodes,
                                    ngraph::op::ParameterVector& parameters)
            {
                auto ng_input = GetInputNode(all_ng_nodes, node, 0);

                // Create a constant tensor populated with the value -1/2.
                // (1/sqrt(x) = x^(-1/2))
                auto shape = ng_input->get_shape();
                std::vector<std::string> constant_values(ngraph::shape_size(shape), "-0.5");

                auto ng_exponent = std::make_shared<ngraph::op::Constant>(ng_input->get_element_type(), shape, constant_values);

                // Raise each element of the input to the power -0.5.
                auto ng_node = std::make_shared<ngraph::op::Power>(ng_input, ng_exponent);

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
                auto clamper = [](int idx, size_t dim, bool inclusive)
                {
                    // if idx is in [-(d-1), d-1], then its same for both inclusive and
                    // exclusive
                    // The first 2 cases breaks down this range
                    if (idx >= 0 && idx <= (static_cast<int>(dim) - 1))
                    {
                        return idx;
                    } 
                    else if (idx < 0 && idx + static_cast<int>(dim) >= 0)
                    {   // careful not to do idx >= -dim
                        // (since dim is unsigned)
                        return idx + static_cast<int>(dim);     // Type casting to int to enable unambiguous auto
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

                auto tf_to_ng = [clamper](int tf_begin_idx, int tf_end_idx, int tf_stride,
                                            size_t dim, bool begin_mask, bool end_mask,
                                            bool shrink_mask)
                {
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
                    int64 clamped_end_idx =
                        clamper(shrink_mask ? clamped_begin_idx + 1 : tf_ignore_end_if_needed,
                                dim, false);

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
                        {   // In the whole of this else: clamped_begin_idx !=
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
                                {   // clamped_begin_idx > clamped_end_idx, tf_stride < 0

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
                    return std::make_tuple(ng_begin_idx, ng_end_idx, std::abs(tf_stride),
                                        needs_reverse);
                };                

                auto extract_bit = [](int bit_mask, int bit_location)
                {
                    return (bit_mask & (1 << bit_location)) != 0;
                };

                auto dim_vec = ng_input->get_shape();
                auto in_rank = dim_vec.size();

                if (begin_vec.size() > in_rank) {
                    std::cerr << "Index out of range using input dim "
                              << begin_vec.size() << "; input has only "
                              << in_rank << " dims";
                    assert(false);
                }

                // TODO/Note/Question: Are begin, end and stride vectors are of equal length

                // begin, end and stride vectors may not have same size as input rank, hence
                // initialize them with 0, dim and 1 respectively
                vector<size_t> ng_begin_vec(in_rank, 0), ng_stride_vec(in_rank, 1);
                vector<size_t> ng_end_vec(dim_vec);
                vector<size_t> ng_needs_reversal(in_rank, 0);   // should have been a
                                                                // vector<bool>, but it is
                                                                // optimized, so tie won't
                                                                // work. Hence using size_t
                for (int dim_idx = 0; dim_idx < begin_vec.size(); dim_idx++) {
                    std::tie(ng_begin_vec[dim_idx], ng_end_vec[dim_idx], ng_stride_vec[dim_idx],
                            ng_needs_reversal[dim_idx]) =
                        tf_to_ng(begin_vec[dim_idx], end_vec[dim_idx], stride_vec[dim_idx],
                                dim_vec[dim_idx], extract_bit(tf_begin_mask, dim_idx),
                                extract_bit(tf_end_mask, dim_idx),
                                extract_bit(tf_shrink_axis_mask, dim_idx));
                }

                // filter out negative stride dimensions
                vector<size_t> neg_strides;
                for (int dim_idx = 0; dim_idx < in_rank; dim_idx++) {
                    if (ng_needs_reversal[dim_idx]) {
                    neg_strides.push_back(dim_idx);
                    }
                }

                // atleast one stride was negative, in which case reverse the input
                if (neg_strides.size() > 0)
                {
                    ng_input = std::make_shared<ngraph::op::Reverse>(ng_input, neg_strides);
                }

                std::shared_ptr<ngraph::Node> ng_strided_slice = std::make_shared<ngraph::op::Slice>(ng_input, ng_begin_vec, ng_end_vec, ng_stride_vec);

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
                                std::cerr << "Trying to shrink specification " << i
                                          << "where tf begin, end, strides are: " << begin_vec[i] << ":"
                                          << end_vec[i] << ":" << stride_vec[i]
                                          << ". nGraph begin, end, stride are: " << ng_begin_vec[i] << ":"
                                          << ng_end_vec[i] << ":" << ng_stride_vec[i]
                                          << ". nGraph's begin and end have difference greater than 1";
                                assert(false);
                            }
                            
                        }
                        shrink_axis_mask >>= 1;
                    }

                    ngraph::Shape ng_final_shape(output_shape);
                    ngraph::AxisVector ng_axis_order(input_shape.size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

                    ng_strided_slice = std::make_shared<ngraph::op::Reshape>(ng_strided_slice, ng_axis_order, ng_final_shape);
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
                auto ng_node = std::make_shared<ngraph::op::Constant>(ngraph::element::i32, ngraph::Shape{}, std::vector<int>{0});

                ng_node->set_name(node.name());
                NamedNodeVector ret{{node.name(), ng_node}};
                return ret;
            }

            const static std::map<const std::string, ConvertFunc> TRANSLATE_OP_MAP{
                {"Abs", TranslateUnaryOp<ngraph::op::Abs>},
                {"Add", TranslateBinaryOp<ngraph::op::Add>},
                {"All", TranslateAllOp},
                {"Assert", TranslateAssertOp},
                {"AvgPool", TranslateAvgPoolOp},
                {"BatchMatMul", TranslateBatchMatMulOp},
                {"BiasAdd", TranslateBiasAddOp},
                {"Cast", TranslateCastOp},
                {"Const", TranslateConstOp},
                {"Conv2D", TranslateConv2DOp},
                {"ConcatV2", TranslateConcatV2Op},
                {"Exp", TranslateUnaryOp<ngraph::op::Exp>},
                {"ExpandDims", TranslateExpandDimsOp},
                {"Fill", TranslateFillOp},
                {"FusedBatchNorm", TranslateFusedBatchNormOp},
                {"FusedBatchNormV2", TranslateFusedBatchNormOp},
                {"GatherV2", TranslateGatherV2Op},
                {"Identity", TranslateIdentityOp},
                {"MatMul", TranslateMatMulOp},
                {"LessEqual", TranslateBinaryOp<ngraph::op::LessEq>},
                {"MaxPool", TranslateMaxPoolOp},
                {"Mean", TranslateMeanOp},
                {"Mul", TranslateBinaryOp<ngraph::op::Multiply>},
                {"OneHot", TranslateOneHotOp},
                {"Pad", TranslatePadOp},
                {"PadV2", TranslatePadV2Op},
                {"Placeholder", TranslateInputOp<ngraph::op::Parameter>},
                {"Pow", TranslateBinaryOp<ngraph::op::Power>},
                {"Relu", TranslateUnaryOp<ngraph::op::Relu>},
                {"Reshape", TranslateReshapeOp},
                {"Rsqrt", TranslateRsqrtOp},
                {"Sigmoid", TranslateSigmoidOp},
                {"Slice", TranslateSliceOp},
                {"Softmax", TranslateSoftmaxOp},
                {"Split", TranslateSplitOp},
                {"SplitV", TranslateSplitVOp},
                {"SquaredDifference", TranslateSquaredDifferenceOp},
                {"Squeeze", TranslateSqueezeOp},
                {"StridedSlice", TranslateStridedSliceOp},
                {"StopGradient", TranslateStopGradientOp},
                {"Sub", TranslateBinaryOp<ngraph::op::Subtract>},
                {"Sum", TranslateSumOp},
                {"Tanh", TranslateUnaryOp<ngraph::op::Tanh>},
                {"Transpose", TranslateTransposeOp}};

            struct InputInfo
            {
                explicit InputInfo(const std::string& node_name,
                                   std::shared_ptr<ngraph::Node> n,
                                   int i)
                    : name(node_name)
                    , node(n)
                    , index(i)
                {
                }
                std::string name;
                std::shared_ptr<ngraph::Node> node;
                int index;
            };

            GraphConvert::GraphConvert(const tensorflow::GraphDef& proto)
                : m_graph_proto{&proto}
            {
                std::cerr << "Converting Tensorflow Graph" << std::endl;
                m_ngraph = std::make_shared<ngraph::Graph>();

                generate_topology();

                uint32_t processed = 0;

                std::vector<InputInfo> inputs;
                while (!topology_.empty())
                {
                    uint32_t node_idx = topology_.front();
                    topology_.pop();
                    ++processed;
                    inputs.clear();
                    const auto& node_proto = proto.node(node_idx);

                    for (auto& input : node_proto.input())
                    {
                        TensorId input_tensor(ParseTensorName(input));
                        int src_index;
                        std::shared_ptr<ngraph::Node> src_node;

                        auto iter = m_ng_node.find(input_tensor.first);
                        if (iter == m_ng_node.end())
                        {
                            std::cerr << "Node " << node_proto.name()
                                      << " has Un-Converted input node: " << input_tensor.first;
                            assert(false);
                        }
                        src_index = input_tensor.second;
                        if (src_index == Graph::kControlSlot)
                        {
                            // TODO: how to handle control edge
                            continue;
                        }
                        src_node = iter->second.at(src_index);
                        inputs.emplace_back(input_tensor.first, src_node, 0);
                    }
                    
                    auto ng_nodes = convert_node(node_proto);
                    for (auto& node : ng_nodes)
                    {
                        m_ng_node[node.first].push_back(node.second);
                        m_ngraph->AddNode(node.second);
                        int input_idx = 0;

                        for (auto& input : node_proto.input())
                        {
                            m_ngraph->AddEdge(inputs[input_idx].node,
                                              inputs[input_idx].index,
                                              node.second,
                                              input_idx);
                            input_idx++;
                            // TODO: ADD CONTROL EDGE;
                        }
                    }

                    for (size_t i = 0; i < tensorflow_node_outputs_[node_idx].size(); ++i)
                    {
                        const int output = tensorflow_node_outputs_[node_idx][i];
                        pending_counts_[output]--;
                        if (pending_counts_[output] == 0)
                        {
                            topology_.push(output);
                        }
                    }
                    if (is_input.find(node_proto.name()) != is_input.end())
                    {
                        for (auto& node : ng_nodes)
                        {
                            m_inputs.emplace_back(node.second);
                        }
                    }
                    if (is_output.find(node_proto.name()) != is_output.end())
                    {
                        for (auto& node : ng_nodes)
                        {
                            m_outputs.emplace_back(node.second);
                        }
                    }
                }
                std::cout << "convert graph done" << endl;

            }

            void GraphConvert::generate_topology()
            {
                const size_t num_nodes = m_graph_proto->node_size();

                for (size_t n = 0; n < num_nodes; ++n)
                {
                    tensorflow_name2nodeIdx_map_[m_graph_proto->node(n).name()] = n;
                }

                pending_counts_.reserve(num_nodes);
                tensorflow_node_outputs_.resize(num_nodes);
                for (size_t n = 0; n < num_nodes; ++n)
                {
                    const auto& node_proto = m_graph_proto->node(n);
                    int pending_count = node_proto.input_size();
                    for (size_t i = 0; i < node_proto.input_size(); ++i)
                    {
                        // TODO: "name:num" or "^name"
                        std::string input_name = node_proto.input(i);
                        TensorId input_tensor(ParseTensorName(input_name));

                        auto iter = tensorflow_name2nodeIdx_map_.find(input_tensor.first);
                        if (iter == tensorflow_name2nodeIdx_map_.end())
                        {
                            std::cerr << "Node " << node_proto.name()
                                      << " has Unknown input node: " << input_name;
                            assert(false);
                        }
                        tensorflow_node_outputs_[iter->second].push_back(n);
                    }
                    if (pending_count == 0)
                    {
                        topology_.push(n);
                    }
                    pending_counts_.push_back(pending_count);
                }

                for (const auto& node_proto : m_graph_proto->node())
                    out_edges_count[node_proto.name()] = 0;
                for (const auto& node_proto : m_graph_proto->node())
                {
                    in_edges_count[node_proto.name()] = node_proto.input_size();
                    for (auto& input : node_proto.input())
                    {
                        ++out_edges_count[input];
                    }
                }

                for (auto& it : in_edges_count)
                    if (it.second == 0)
                        is_input.insert(it.first);
                for (auto& it : out_edges_count)
                    if (it.second == 0)
                        is_output.insert(it.first);
            }

            NamedNodeVector GraphConvert::convert_node(const tensorflow::NodeDef& node)
            {
                auto func = TRANSLATE_OP_MAP.find(node.op());
                if (func != TRANSLATE_OP_MAP.end())
                {
                    return func->second(node, m_ng_node, m_parameters);
                }
                else
                {
                    std::cerr << "Unsupport operator: " << node.op() << std::endl;
                    return NamedNodeVector{};
                }
            }

            std::vector<std::shared_ptr<ngraph::Function>> GraphConvert::get_outputs()
            {
                std::vector<std::shared_ptr<Function>> output_functions;
                for (const auto& output : m_outputs)
                {
                    output_functions.emplace_back(
                        std::make_shared<ngraph::Function>(output, m_parameters));
                }
                return output_functions;
            }
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace ngraph
//----------------------------------------------------------------------------------------------
