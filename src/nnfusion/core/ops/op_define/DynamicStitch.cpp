// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(DynamicStitch)
    .attr<int>("N")
    .attr<ngraph::op::OpConfig::any>("T")
    .infershape([](ngraph::op::GenericOp& target_op) -> void {
        size_t input_size = target_op.get_input_size();
        int num_partitions = target_op.localOpConfig.getRoot()["N"];
        enforce(num_partitions * 2 == input_size);

        bool all_indices_constant = true;
        int64_t max_index = 0;
        ngraph::Shape output_shape;
        ngraph::element::Type type;
        std::vector<std::vector<int64_t>> indices_inputs;
        for (int i = 0; i < num_partitions; ++i)
        {
            std::shared_ptr<Node> indices_node = target_op.inputs[i];
            if (indices_node->description() == "Constant")
            {
                auto ng_constant_op = std::dynamic_pointer_cast<ngraph::op::Constant>(indices_node);
                auto ng_element_type = ng_constant_op->get_element_type();
                enforce(ng_element_type == ngraph::element::i32 ||
                        ng_element_type == ngraph::element::i64);
                std::vector<int64_t> values;
                if (ng_element_type == ngraph::element::i32)
                {
                    std::vector<int32_t> values_int32 = ng_constant_op->get_vector<int32_t>();
                    values.insert(values.begin(), values_int32.begin(), values_int32.end());
                }
                else
                {
                    values = ng_constant_op->get_vector<int64_t>();
                }

                indices_inputs.push_back(values);
                for (size_t i = 0; i < values.size(); i++)
                {
                    if (values[i] > max_index)
                        max_index = values[i];
                }
            }
            else
            {
                all_indices_constant = false;
                enforce(false) << "currently we do not support dynamic tensor shape, input_node="
                               << indices_node->description();
            }
            auto& indices_shape = target_op.get_input_shape(i);
            auto& data_shape = target_op.get_input_shape(i + num_partitions);
            type = target_op.get_input_element_type(i + num_partitions);

            // calculate the sub-shape
            int64_t start = indices_shape.size();
            const int64_t rank = data_shape.size();
            int64_t end = rank;
            if (start > rank)
                start = rank;
            ngraph::Shape dims;
            dims.push_back(max_index + 1);
            for (int i = start; i < end; i++)
            {
                dims.push_back(data_shape[i]);
            }
            output_shape = dims;
        }
        target_op.localOpConfig.attr("indices_inputs", indices_inputs);
        target_op.set_output_type(0, type, output_shape);
    });
