// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(BatchMatMul)
    .attr<ngraph::op::OpConfig::any>("adj_x", {{"b", false}})
    .attr<ngraph::op::OpConfig::any>("adj_y", {{"b", false}})
    .constrait([](const ngraph::op::OpConfig::any& config) -> bool {
        if (!config["adj_x"]["b"].is_boolean())
            return false;
        if (!config["adj_y"]["b"].is_boolean())
            return false;
        return true;
    })
    .infershape([](ngraph::op::GenericOp& target_op) -> void {
        assert(target_op.get_input_size() == 2);

        const ngraph::Shape& input_shape_0 = target_op.get_input_shape(0);
        const ngraph::Shape& input_shape_1 = target_op.get_input_shape(1);
        ngraph::Shape output_shape_0;

        assert(input_shape_0.size() == input_shape_1.size());
        assert(target_op.get_input_element_type(0) == target_op.get_input_element_type(1));

        for (int i = 0; i < input_shape_0.size() - 2; i++)
        {
            assert(input_shape_0[i] == input_shape_1[i]);
            output_shape_0.push_back(input_shape_0[i]);
        }

        int m0 = input_shape_0[input_shape_0.size() - 2],
            n0 = input_shape_0[input_shape_0.size() - 1];
        int m1 = input_shape_1[input_shape_1.size() - 2],
            n1 = input_shape_1[input_shape_1.size() - 1];

        bool trans_A = target_op.localOpConfig.getRoot()["adj_x"]["b"];
        bool trans_B = target_op.localOpConfig.getRoot()["adj_y"]["b"];

        if (!trans_A && !trans_B)
            assert(m1 == n0), output_shape_0.push_back(m0), output_shape_0.push_back(n1);
        else if (!trans_A && trans_B)
            assert(n0 == n1), output_shape_0.push_back(m0), output_shape_0.push_back(m1);
        else if (trans_A && !trans_B)
            assert(m0 == m1), output_shape_0.push_back(n0), output_shape_0.push_back(n1);
        else // trans_A && trans_B
            assert(m0 == n1), output_shape_0.push_back(n0), output_shape_0.push_back(m1);
        target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
    });
