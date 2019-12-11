// Microsoft (c) 2019, NNFusion Team

#include <set>
#include <sstream>

#include "batch_norm.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/operators/util/validation_util.hpp"

using namespace std;
using namespace nnfusion::op;

BatchNormInference::BatchNormInference(double eps)
    : Op("BatchNormInference")
    , m_epsilon(eps)
{
}

BatchNormTraining::BatchNormTraining(double eps)
    : Op("BatchNormTraining")
    , m_epsilon(eps)
{
}

void BatchNormInference::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    ngraph::element::Type result_et;
    ngraph::PartialShape result_batch_shape;
    ngraph::PartialShape result_channel_shape; // unused here

    std::tie(result_et, result_batch_shape, result_channel_shape) =
        infer_batch_norm_forward(this,
                                 gnode->get_input_element_type(INPUT_DATA),
                                 gnode->get_input_element_type(INPUT_GAMMA),
                                 gnode->get_input_element_type(INPUT_BETA),
                                 gnode->get_input_element_type(INPUT_MEAN),
                                 gnode->get_input_element_type(INPUT_VARIANCE),
                                 gnode->get_input_partial_shape(INPUT_DATA),
                                 gnode->get_input_partial_shape(INPUT_GAMMA),
                                 gnode->get_input_partial_shape(INPUT_BETA),
                                 gnode->get_input_partial_shape(INPUT_MEAN),
                                 gnode->get_input_partial_shape(INPUT_VARIANCE));

    set_output_type_and_shape(gnode, 0, result_et, result_batch_shape);
}

void BatchNormTraining::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    ngraph::element::Type result_et;
    ngraph::PartialShape result_batch_shape;
    ngraph::PartialShape result_channel_shape;

    std::tie(result_et, result_batch_shape, result_channel_shape) =
        infer_batch_norm_forward(this,
                                 gnode->get_input_element_type(INPUT_DATA),
                                 gnode->get_input_element_type(INPUT_GAMMA),
                                 gnode->get_input_element_type(INPUT_BETA),
                                 gnode->get_input_partial_shape(INPUT_DATA),
                                 gnode->get_input_partial_shape(INPUT_GAMMA),
                                 gnode->get_input_partial_shape(INPUT_BETA));

    gnode->set_output_size(3);
    set_output_type_and_shape(gnode, 0, result_et, result_batch_shape);
    set_output_type_and_shape(gnode, 1, result_et, result_channel_shape);
    set_output_type_and_shape(gnode, 2, result_et, result_channel_shape);
}

BatchNormTrainingBackprop::BatchNormTrainingBackprop(double eps)
    : Op("BatchNormTrainingBackprop")
    , m_epsilon(eps)

{
}

void BatchNormTrainingBackprop::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    ngraph::PartialShape input_and_delta_shape{gnode->get_input_partial_shape(INPUT_DATA)};

    OP_VALIDATION(this,
                  PartialShape::merge_into(input_and_delta_shape,
                                           gnode->get_input_partial_shape(INPUT_DELTA)))
        << "Shape of delta does not match the shape of the input data (input data shape: "
        << gnode->get_input_partial_shape(INPUT_DATA)
        << ", delta shape: " << gnode->get_input_partial_shape(INPUT_DELTA) << ").";

    ngraph::element::Type input_and_delta_et;

    OP_VALIDATION(this,
                  ngraph::element::Type::merge(input_and_delta_et,
                                               gnode->get_input_element_type(INPUT_DATA),
                                               gnode->get_input_element_type(INPUT_DELTA)))
        << "Element type for input (" << gnode->get_input_element_type(INPUT_DATA)
        << ") does not match element type for delta (" << gnode->get_input_element_type(INPUT_DATA)
        << ").";

    ngraph::element::Type result_et;
    ngraph::PartialShape result_batch_shape;
    ngraph::PartialShape result_channel_shape;

    std::tie(result_et, result_batch_shape, result_channel_shape) =
        infer_batch_norm_forward(this,
                                 input_and_delta_et,
                                 gnode->get_input_element_type(INPUT_GAMMA),
                                 gnode->get_input_element_type(INPUT_BETA),
                                 gnode->get_input_element_type(INPUT_MEAN),
                                 gnode->get_input_element_type(INPUT_VARIANCE),
                                 input_and_delta_shape,
                                 gnode->get_input_partial_shape(INPUT_GAMMA),
                                 gnode->get_input_partial_shape(INPUT_BETA),
                                 gnode->get_input_partial_shape(INPUT_MEAN),
                                 gnode->get_input_partial_shape(INPUT_VARIANCE));

    gnode->set_output_size(3);
    set_output_type_and_shape(gnode, 0, result_et, result_batch_shape);
    set_output_type_and_shape(gnode, 1, result_et, result_channel_shape);
    set_output_type_and_shape(gnode, 2, result_et, result_channel_shape);
}