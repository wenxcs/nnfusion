// Microsoft (c) 2019, NNFusion Team

#include "ngraph/serializer.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(OneHot)
    .attr<int>("axis", -1)
    .attr<int>("depth")
    .attr<ngraph::op::OpConfig::any>("T")
    .attr<ngraph::op::OpConfig::any>("off_value", 1.0f)
    .attr<ngraph::op::OpConfig::any>("on_value", 0.0f)
    .infershape([](ngraph::op::GenericOp& target_op) -> void {
        assert(1 == target_op.get_input_size());
        auto& shape_0 = target_op.get_input_shape(0);
        int depth = target_op.localOpConfig.getRoot()["depth"];
        int axis = target_op.localOpConfig.getRoot()["axis"];
        std::string t_str = target_op.localOpConfig.getRoot()["T"];

        size_t bitwidth = 0;
        bool is_real = false;
        bool is_signed = false;
        bool is_quantized = false;
        string c_type_string = "";
        for (const element::Type* t : element::Type::get_known_types())
        {
            if (t->c_type_string() == t_str)
            {
                bitwidth = t->bitwidth();
                is_real = t->is_real();
                is_signed = t->is_signed();
                is_quantized = t->is_quantized();
                c_type_string = t->c_type_string();
                break;
            }
        }
        ngraph::element::Type type =
            element::Type(bitwidth, is_real, is_signed, is_quantized, c_type_string);

        if (axis == -1)
            axis = shape_0.size() - 1;
        ngraph::Shape output_shape_0;
        for (int i = 0; i <= axis; ++i)
            output_shape_0.push_back(shape_0[i]);
        output_shape_0.push_back(depth);
        for (int i = axis + 1; i < shape_0.size(); ++i)
            output_shape_0.push_back(shape_0[i]);
        target_op.set_output_type(0, type, output_shape_0);
    });
