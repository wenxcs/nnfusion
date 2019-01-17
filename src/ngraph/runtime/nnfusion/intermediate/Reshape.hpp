// Microsoft (c) 2019, Wenxiang
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_op.hpp"

using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            namespace intermediate
            {
                class Reshape : public IntermediateOP
                {
                private:
                    Shape arg_shape;
                    size_t arg_rank;
                    Shape result_shape;
                    AxisVector input_order;
                    size_t result_shape_product;
                    bool isMemCpy = false;

                public:
                    static std::shared_ptr<IntermediateOP> translate(TRANS_ARGS)
                    {
                        std::shared_ptr<Reshape> inter_op(new Reshape());
                        inter_op->isTranslated = false;
                        if (out[0].get_size() == 0)
                        {
                            return inter_op;
                        }
                        auto reshape = static_cast<const op::Reshape*>(node);

                        if (out[0].get_name() == args[0].get_name())
                        {
                            return inter_op;
                        }

                        // Store the info
                        std::cout << args[0].get_name() << std::endl;
                        inter_op->arg_shape = args[0].get_shape();
                        inter_op->arg_rank = inter_op->arg_shape.size();
                        inter_op->result_shape = out[0].get_shape();
                        inter_op->input_order = reshape->get_input_order();

                        std::cout << inter_op->arg_shape << " " << inter_op->result_shape
                                  << std::endl;

                        size_t result_shape_product = shape_size(inter_op->result_shape);

                        //for a zero-size tensor, or change from 1^m shape to 1^n shape, just do a copy
                        if (!reshape->get_is_transpose() || result_shape_product < 2)
                        {
                            inter_op->isMemCpy = true;
                            inter_op->isTranslated = true;
                            return inter_op;
                        }

                        //combine inordered dimensons after reorder in shape, update output shape and input order
                        Shape in_order_map(inter_op->arg_rank, 0);
                        for (int i = 0; i < inter_op->arg_rank - 1; i++)
                        {
                            if (static_cast<int64_t>(inter_op->input_order[i + 1]) -
                                    static_cast<int64_t>(inter_op->input_order[i]) ==
                                1)
                            {
                                in_order_map[inter_op->input_order[i]] = 1;
                            }
                        }

                        Shape combine_arg_shape;
                        Shape combine_idx_map(inter_op->arg_rank, 0);
                        Shape combine_input_order;
                        size_t shape_i = 1;
                        size_t combine_rank = 0;
                        for (int i = 0; i < inter_op->arg_rank; i++)
                        {
                            if (in_order_map[i] == 1)
                            {
                                shape_i *= inter_op->arg_shape[i];
                            }
                            else
                            {
                                combine_arg_shape.push_back(shape_i * inter_op->arg_shape[i]);
                                shape_i = 1;
                                combine_idx_map[i] = combine_rank++;
                            }
                        }

                        for (int i = 0; i < inter_op->arg_rank; i++)
                        {
                            if (in_order_map[inter_op->input_order[i]] == 0)
                            {
                                combine_input_order.push_back(
                                    combine_idx_map[inter_op->input_order[i]]);
                            }
                        }

                        //eleminate dimenson size = 1, update input order and output shape
                        Shape new_arg_shape;
                        Shape new_result_shape;
                        Shape new_idx_map(combine_rank, 0);
                        Shape new_input_order;
                        size_t new_rank = 0;
                        for (int i = 0; i < combine_rank; i++)
                        {
                            if (combine_arg_shape[i] != 1)
                            {
                                new_arg_shape.push_back(combine_arg_shape[i]);
                                new_idx_map[i] = new_rank++;
                            }
                        }
                        for (int i = 0; i < combine_rank; i++)
                        {
                            if (combine_arg_shape[combine_input_order[i]] != 1)
                            {
                                new_input_order.push_back(new_idx_map[combine_input_order[i]]);
                            }
                        }
                        for (int i = 0; i < new_rank; i++)
                        {
                            new_result_shape.push_back(new_arg_shape[new_input_order[i]]);
                        }

                        inter_op->isTranslated = true;
                        return inter_op;
                    }
                };
            }
        }
    }
}