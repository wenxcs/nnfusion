// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(BatchMatMul)
    .attr<nnfusion::op::OpConfig::any>("adj_x", {{"b", false}})
    .attr<nnfusion::op::OpConfig::any>("adj_y", {{"b", false}})
    .constrait([](const nnfusion::op::OpConfig::any& config) -> bool {
        if (!config["adj_x"]["b"].is_boolean())
            return false;
        if (!config["adj_y"]["b"].is_boolean())
            return false;
        return true;
    })
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        CHECK(gnode->get_input_size() == 2);

        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);
        nnfusion::Shape output_shape_0;

        CHECK(input_shape_0.size() == input_shape_1.size());
        CHECK(gnode->get_input_element_type(0) == gnode->get_input_element_type(1));

        for (int i = 0; i < input_shape_0.size() - 2; i++)
        {
            CHECK(input_shape_0[i] == input_shape_1[i]);
            output_shape_0.push_back(input_shape_0[i]);
        }

        int m0 = input_shape_0[input_shape_0.size() - 2],
            n0 = input_shape_0[input_shape_0.size() - 1];
        int m1 = input_shape_1[input_shape_1.size() - 2],
            n1 = input_shape_1[input_shape_1.size() - 1];

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        bool trans_A = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
        bool trans_B = generic_op->localOpConfig.getRoot()["adj_y"]["b"];

        if (!trans_A && !trans_B)
            CHECK(m1 == n0), output_shape_0.push_back(m0), output_shape_0.push_back(n1);
        else if (!trans_A && trans_B)
            CHECK(n0 == n1), output_shape_0.push_back(m0), output_shape_0.push_back(m1);
        else if (trans_A && !trans_B)
            CHECK(m0 == m1), output_shape_0.push_back(n0), output_shape_0.push_back(n1);
        else // trans_A && trans_B
            CHECK(m0 == n1), output_shape_0.push_back(n0), output_shape_0.push_back(m1);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    })
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        CHECK(gnode->get_input_size() == 2);

        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);
        nnfusion::Shape output_shape = gnode->get_output_shape(0);

        CHECK(input_shape_0.size() == input_shape_1.size());
        CHECK(gnode->get_input_element_type(0) == gnode->get_input_element_type(1));
        CHECK(input_shape_0[0] == input_shape_1[0] || input_shape_0[0] == 1 ||
              input_shape_1[0] == 1);

        int m0 = input_shape_0[input_shape_0.size() - 2],
            n0 = input_shape_0[input_shape_0.size() - 1];

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        bool trans_A = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
        bool trans_B = generic_op->localOpConfig.getRoot()["adj_y"]["b"];
        int k;
        std::string input0_expr = std::string("args(\"input0\")[");
        std::string input1_expr = std::string("args(\"input1\")[");
        if (input_shape_0[0] == 1)
        {
            input0_expr += "1";
        }
        else
        {
            input0_expr += "b";
        }
        if (input_shape_1[0] == 1)
        {
            input1_expr += "1";
        }
        else
        {
            input1_expr += "b";
        }
        if (!trans_A)
        {
            k = n0;
            input0_expr += ", i, k]";
        }
        else
        {
            k = m0;
            input0_expr += ", k, i]";
        }

        if (!trans_B)
        {
            input1_expr += ", k, j]";
        }
        else
        {
            input1_expr += ", j, k]";
        }

        auto expression = op::create_code_from_template(
            R"( - input("input0", @input_shape_0@); input("input1", @input_shape_1@); k = loop(@k@); output(@output_shape@, lambda b, i, j: tvm.sum(@input0_expr@ * @input1_expr@, axis=k)); )",
            {{"input_shape_0", vector_to_string(input_shape_0)},
             {"input_shape_1", vector_to_string(input_shape_1)},
             {"output_shape", vector_to_string(output_shape)},
             {"k", k},
             {"input0_expr", input0_expr},
             {"input1_expr", input1_expr}});

        return expression;
    });
