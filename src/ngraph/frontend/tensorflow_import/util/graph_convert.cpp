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
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/slice.hpp"
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
                if (adj_x)
                {
                    ng_lhs = ngraph::builder::numpy_transpose(ng_lhs, ngraph::AxisVector{1, 0});
                }
                if (adj_y)
                {
                    ng_rhs = ngraph::builder::numpy_transpose(ng_rhs, ngraph::AxisVector{1, 0});
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

            const static std::map<const std::string, ConvertFunc> TRANSLATE_OP_MAP{
                {"Abs", TranslateUnaryOp<ngraph::op::Abs>},
                {"Add", TranslateBinaryOp<ngraph::op::Add>},
                {"AvgPool", TranslateAvgPoolOp},
                {"BiasAdd", TranslateBiasAddOp},
                {"Cast", TranslateCastOp},
                {"Const", TranslateConstOp},
                {"Conv2D", TranslateConv2DOp},
                {"ConcatV2", TranslateConcatV2Op},
                {"Exp", TranslateUnaryOp<ngraph::op::Exp>},
                {"Fill", TranslateFillOp},
                {"FusedBatchNorm", TranslateFusedBatchNormOp},
                {"FusedBatchNormV2", TranslateFusedBatchNormOp},
                {"Identity", TranslateIdentityOp},
                {"MatMul", TranslateMatMulOp},
                {"BatchMatMul", TranslateBatchMatMulOp},
                {"MaxPool", TranslateMaxPoolOp},
                {"Mean", TranslateMeanOp},
                {"Mul", TranslateBinaryOp<ngraph::op::Multiply>},
                {"Pad", TranslatePadOp},
                {"PadV2", TranslatePadV2Op},
                {"Placeholder", TranslateInputOp<ngraph::op::Parameter>},
                {"Pow", TranslateBinaryOp<ngraph::op::Power>},
                {"Relu", TranslateUnaryOp<ngraph::op::Relu>},
                {"Reshape", TranslateReshapeOp},
                {"Sigmoid", TranslateSigmoidOp},
                {"Slice", TranslateSliceOp},
                {"Split", TranslateSplitOp},
                {"SplitV", TranslateSplitVOp},
                {"Sub", TranslateBinaryOp<ngraph::op::Subtract>},
                {"Sum", TranslateSumOp},
                {"Tanh", TranslateUnaryOp<ngraph::op::Tanh>}};

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
                    //std::cout << node_proto.DebugString() << std::endl;

                    size_t i = 0;
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

                        src_node = iter->second.at(src_index);
                        inputs.emplace_back(input_tensor.first, src_node, 0);
                        i++;
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
