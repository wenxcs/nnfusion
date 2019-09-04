// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/ops/generic_op.hpp"

REGISTER_OP(InvertPermutation)
    .attr<ngraph::op::OpConfig::any>("T")
    .infershape([](ngraph::op::GenericOp& target_op) -> void {
        // enforce is like assert, but when thing goes wrong, it will print error message.
        enforce(target_op.get_input_size() == 1)
            << "Only one input is allowed for the InvertPermutation operator";

        auto& shape_0 = target_op.get_input_shape(0);
        enforce(shape_0.size() == 1) << "The input only can take a 1-D integer tensor";

        auto ng_op = target_op.get_argument(0);
        if (ng_op->description() == "Constant")
        {
            std::unordered_map<int, int> element_records;
            auto input_vector =
                std::dynamic_pointer_cast<ngraph::op::Constant>(ng_op)->get_vector<int>();

            for (int i = 0; i < input_vector.size(); i++)
            {
                enforce(input_vector[i] >= 0 && input_vector[i] < input_vector.size())
                    << "The elements for InvertPermutation's inputs must between 0 to n-1";
                element_records[input_vector[i]]++;
                enforce(element_records[input_vector[i]] == 1)
                    << "The frequency of a number in InvertPermutation's inputs cannot above 1";
            }
            enforce(element_records.size() == (input_vector.size() - 1))
                << "The input vector must contain all number between 0 to n-1";
        }

        ngraph::Shape output_shape_0(shape_0);
        target_op.set_output_type(0, target_op.get_input_element_type(0), output_shape_0);
    });