// Microsoft (c) 2019, Wenxiang
#include "reshape.hpp"
#include "noop.hpp"
#include "result.hpp"

using namespace nnfusion::ir;

Reshape::Reshape(shared_ptr<Node> node)
    : Operator(node)
{
    enforce(out[0].get_size() > 0) << "Invalid output shape for Reshape.";
    reshape = static_pointer_cast<ngraph::op::Reshape>(node);

    //Noop
    if (out[0].get_name() == args[0].get_name())
    {
        LOG_INFO << "Same input and output tensor." << endl;
        return;
    }

    arg_shape = args[0].get_shape();
    arg_rank = arg_shape.size();
    result_shape = out[0].get_shape();
    input_order = reshape->get_input_order();
    size_t result_shape_product = shape_size(result_shape);

    //Result OP
    //for a zero-size tensor, or change from 1^m shape to 1^n shape, just do a copy
    if (!reshape->get_is_transpose() || result_shape_product < 2)
    {
        LOG_INFO << "No need for zero-size or 1-d tensor reshape." << endl;
        return;
    }

    //combine inordered dimensons after reorder in shape, update output shape and input order
    Shape in_order_map(arg_rank, 0);
    for (int i = 0; i < arg_rank - 1; i++)
    {
        if (static_cast<int64_t>(input_order[i + 1]) - static_cast<int64_t>(input_order[i]) == 1)
        {
            in_order_map[input_order[i]] = 1;
        }
    }

    Shape combine_arg_shape;
    Shape combine_idx_map(arg_rank, 0);
    Shape combine_input_order;
    size_t shape_i = 1;
    size_t combine_rank = 0;
    for (int i = 0; i < arg_rank; i++)
    {
        if (in_order_map[i] == 1)
        {
            shape_i *= arg_shape[i];
        }
        else
        {
            combine_arg_shape.push_back(shape_i * arg_shape[i]);
            shape_i = 1;
            combine_idx_map[i] = combine_rank++;
        }
    }

    for (int i = 0; i < arg_rank; i++)
    {
        if (in_order_map[input_order[i]] == 0)
        {
            combine_input_order.push_back(combine_idx_map[input_order[i]]);
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

    arg_shape = new_arg_shape;
    arg_rank = arg_shape.size();
    result_shape = new_result_shape;
    input_order = new_input_order;
}

bool Reshape::isMemcpy()
{
    size_t result_shape_product = shape_size(result_shape);
    //for a zero-size tensor, or change from 1^m shape to 1^n shape, just do a copy
    //or no layout change
    if (!reshape->get_is_transpose() || result_shape_product < 2 ||
        is_sorted(input_order.begin(), input_order.end()))
        return true;
    return false;
}

bool Reshape::isNoop()
{
    //for a zero-size tensor, or change from 1^m shape to 1^n shape, just do a copy
    //or no layout change
    if (out[0].get_name() == args[0].get_name())
        return true;
    return false;
}

Operator_p Reshape::translate(shared_ptr<Node> node)
{
    create_ptr(Reshape, inter_op, node);
    /* //<TODO> Support this in future
    if (inter_op->isNoop())
    {
        LOG_INFO << "Translate this Reshape to Noop" << endl;
        create_ptr(Noop, nop, node);
        return nop;
    }
    if (inter_op->isMemcpy())
    {
        LOG_INFO << "Translate this Reshape to Result(memcopy)" << endl;
        create_ptr(Result, res, node);
        return res;
    }
    */
    return inter_op;
}