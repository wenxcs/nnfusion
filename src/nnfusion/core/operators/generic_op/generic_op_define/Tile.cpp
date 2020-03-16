// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Tile)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() == 2);
        auto& input_shape_0 = gnode->get_input_shape(0);
        auto ng_op = gnode->get_in_edge(1)->get_src();
        NNFUSION_CHECK(ng_op->get_op_type() == "Constant")
            << "We only accept the Tile input \"multiples\" as Constant.";
        ///\todo multiples must be int32 or int64, we use int32 in this case, currently we ignore int64
        auto multiples = std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                             ->get_vector<int64_t>();
        NNFUSION_CHECK(input_shape_0.size() == multiples.size());
        nnfusion::Shape output_shape_0(multiples.size());
        for (int i = 0; i < multiples.size(); i++)
            output_shape_0[i] = multiples[i] * input_shape_0[i];
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    })
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        nnfusion::Shape input_shape = gnode->get_input_shape(0);
        nnfusion::Shape output_shape = gnode->get_output_shape(0);

        auto ng_op = gnode->get_in_edge(1)->get_src();
        NNFUSION_CHECK(ng_op->get_op_type() == "Constant")
            << "We only accept the Tile input \"multiples\" as Constant.";
        ///\todo multiples must be int32 or int64, we use int32 in this case, currently we ignore int64
        auto multiples = std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                             ->get_vector<int64_t>();

        auto expression = op::create_code_from_template(
            R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.tile(args("input0"), @multiples@)); )",
            {{"input_shape", vector_to_string(input_shape)},
             {"output_shape", vector_to_string(output_shape)},
             {"multiples", vector_to_string(multiples)}});

        return expression;
    });
