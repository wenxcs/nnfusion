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
        NNFUSION_CHECK(gnode->get_input_size() == 2);

        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);
        nnfusion::Shape output_shape_0;

        NNFUSION_CHECK(input_shape_0.size() == input_shape_1.size());
        NNFUSION_CHECK(gnode->get_input_element_type(0) == gnode->get_input_element_type(1));

        for (int i = 0; i < input_shape_0.size() - 2; i++)
        {
            NNFUSION_CHECK(input_shape_0[i] == input_shape_1[i]);
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
            NNFUSION_CHECK(m1 == n0), output_shape_0.push_back(m0), output_shape_0.push_back(n1);
        else if (!trans_A && trans_B)
            NNFUSION_CHECK(n0 == n1), output_shape_0.push_back(m0), output_shape_0.push_back(m1);
        else if (trans_A && !trans_B)
            NNFUSION_CHECK(m0 == m1), output_shape_0.push_back(n0), output_shape_0.push_back(n1);
        else // trans_A && trans_B
            NNFUSION_CHECK(m0 == n1), output_shape_0.push_back(n0), output_shape_0.push_back(m1);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    })
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        NNFUSION_CHECK(gnode->get_input_size() == 2);

        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);
        nnfusion::Shape output_shape = gnode->get_output_shape(0);

        NNFUSION_CHECK(input_shape_0.size() == input_shape_1.size());
        NNFUSION_CHECK(gnode->get_input_element_type(0) == gnode->get_input_element_type(1));
        NNFUSION_CHECK(input_shape_0[0] == input_shape_1[0] || input_shape_0[0] == 1 ||
                       input_shape_1[0] == 1);

        int m0 = input_shape_0[input_shape_0.size() - 2],
            n0 = input_shape_0[input_shape_0.size() - 1];

        int batch = 1;
        for (int i = 0; i < input_shape_0.size() - 2; ++i)
            batch *= input_shape_0[i];

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        bool trans_A = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
        bool trans_B = generic_op->localOpConfig.getRoot()["adj_y"]["b"];

        int N = trans_A ? input_shape_0[input_shape_0.size() - 1]
                        : input_shape_0[input_shape_0.size() - 2];
        int K = trans_A ? input_shape_0[input_shape_0.size() - 2]
                        : input_shape_0[input_shape_0.size() - 1];
        int M = trans_B ? input_shape_1[input_shape_1.size() - 2]
                        : input_shape_1[input_shape_1.size() - 1];

        std::string input0 = trans_A ? "args(\"input0\")[b, k, n]" : "args(\"input0\")[b, n, k]";
        std::string input1 = trans_B ? "args(\"input1\")[b, m, k]" : "args(\"input1\")[b, k, m]";

        std::vector<int> shape0 = {batch, N, K};
        if (trans_A)
            std::swap(shape0[1], shape0[2]);
        std::vector<int> shape1 = {batch, K, M};
        if (trans_B)
            std::swap(shape1[1], shape1[2]);
        std::vector<int> outshape = {batch, N, M};

        auto expression = op::create_code_from_template(
            R"( - input("input0", @input_shape_0@); input("input1", @input_shape_1@); k = loop(@k@); output(@output_shape@, lambda b, n, m: tvm.sum(@input0_expr@ * @input1_expr@, axis=k)); )",
            {{"input_shape_0", vector_to_string(shape0)},
             {"input_shape_1", vector_to_string(shape1)},
             {"output_shape", vector_to_string(outshape)},
             {"k", K},
             {"input0_expr", input0},
             {"input1_expr", input1}});

        return expression;
    });
