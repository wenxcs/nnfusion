// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Sum)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Sum>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();
        auto axes = _op->get_reduction_axes();
        auto in_shape = curr->get_input_shape(0);

        std::vector<int> ordered_axes(axes.begin(), axes.end());
        std::sort(ordered_axes.begin(), ordered_axes.end());

        auto product = [&](int start, int stop) -> size_t {
            if (start < 0)
                start += (int)in_shape.size();
            if (stop <= 0)
                stop += (int)in_shape.size();
            size_t base = 1;
            for (int i = start; i < stop; ++i)
                base *= in_shape[i];
            return base;
        };

        // ReduceAll
        if (!ordered_axes.size())
        {
            return op::create_code_from_template(
                R"( - einstein_v2("output0[N] +=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [1, @num_elements@]}});  ## :@ plan/reduce_sum_v1)",
                {
                    {"num_elements", product(0, 0)},
                });
        }
        // ReduceHigh
        if (ordered_axes.front() == 0 && ordered_axes.back() == ordered_axes.size() - 1)
        {
            return op::create_code_from_template(
                R"( - einstein_v2("output0[C] +=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [@sample@, @batch@]}});  ## :@ plan/reduce_sum_v1)",
                {
                    {"sample", product(0, ordered_axes.size())},
                    {"batch", product(ordered_axes.size(), 0)},
                });
        }
        // ReduceLow
        if (ordered_axes.front() == in_shape.size() - ordered_axes.size() &&
            ordered_axes.back() == in_shape.size() - 1)
        {
            return op::create_code_from_template(
                R"( - einstein_v2("output0[N] +=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [@batch@, @sample@]}});  ## :@ plan/reduce_sum_v1)",
                {
                    {"sample", product(0, ordered_axes.size())},
                    {"batch", product(ordered_axes.size(), 0)},
                });
        }

        auto input_shape = curr->get_input_shape(0);

        int min_axis = axes.size() + 1;
        if (axes.size() == 0)
        {
            min_axis = 0;
        }
        else
        {
            for (auto& axis : axes)
                min_axis = min(min_axis, (int)axis);
        }

        if (input_shape.size() - axes.size() == min_axis || axes.size() == 0)
        {
            int batch = 1, sample = 1;
            for (int i = 0; i < min_axis; ++i)
                batch *= input_shape[i];
            for (int i = min_axis; i < input_shape.size(); ++i)
                sample *= input_shape[i];

            return op::create_code_from_template(
                " - input(\"input0\", [@batch@, @sample@]); output([@batch@], "
                "topi=topi.sum(args(\"input0\"), axis=@axis@, keepdims=True));",
                {{"batch", batch}, {"sample", sample}, {"axis", axes.size() != 0 ? "1" : "None"}});
        }
        else
        {
            return op::create_code_from_template(
                " - input(\"input0\", @input_shape@); output(@output_shape@, "
                "topi=topi.sum(args(\"input0\"), axis=@axis@, keepdims=True));",
                {{"input_shape", vector_to_string(input_shape)},
                 {"output_shape", vector_to_string(curr->get_output_shape(0))},
                 {"axis", vector_to_string(axes)}});
        }

    });
