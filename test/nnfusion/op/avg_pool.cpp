// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for AvgPool
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/op/avg_pool.hpp"
#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace inventory
    {
        // We use float by default
        template <>
        shared_ptr<op::AvgPool> create_object(int option)
        {
            switch (option)
            {
            case 0:
            {
                // From testcase: avg_pool_2d_1channel_1image_padded_do_not_include_in_computation
                // Parameter
                Shape shape_a{1, 1, 3, 3};
                Shape window_shape{2, 2};
                auto window_movement_strides = Strides{1, 1};
                Shape padding_below{1, 1};
                Shape padding_above{1, 1};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);

                // AvgPool
                auto op = make_shared<op::AvgPool>(
                    A, window_shape, window_movement_strides, padding_below, padding_above, false);

                // Template for print debug info of input/output tensor
                /*
                auto tensor = op->get_inputs()[0].get_output().get_tensor_ptr();
                cout<<tensor->get_name()<<endl;
                cout<<tensor->get_shape()<<endl;
                cout<<tensor->get_element_type()<<endl;

                auto& v = op->get_outputs();
                auto out = v[0].get_tensor_ptr();
                cout<<out->get_name()<<endl;
                cout<<out->get_shape()<<endl;
                cout<<out->get_element_type()<<endl;
                */
                return op;
            };
            case 1:
            {
                Shape shape_a{2, 2, 14};
                Shape shape_r{2, 2, 12};
                Shape window_shape{3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);

                auto op = make_shared<op::AvgPool>(A, window_shape);

                return op;
            }
            }
            return nullptr;
        }

        template <>
        vector<float> generate_input<op::AvgPool, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                // image 1x1x3x3
                return vector<float>{0, 1, 0, 0, 3, 2, 2, 0, 0};
            }
            case 1:
            {
                return vector<float>{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0,
                                     0, 2, 3, 0, 1, 2, 0, 1, 0, 0, 2, 1, 1, 0, 0, 0, 2, 0, 1,
                                     0, 0, 1, 2, 2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0};
            }
            }
        }

        template <>
        vector<float> generate_output<op::AvgPool, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                // tensor 1x1x4x4
                return vector<float>{0.0f / 1,
                                     1.0f / 2,
                                     1.0f / 2,
                                     0.0f / 1,
                                     0.0f / 2,
                                     4.0f / 4,
                                     6.0f / 4,
                                     2.0f / 2,
                                     2.0f / 2,
                                     5.0f / 4,
                                     5.0f / 4,
                                     2.0f / 2,
                                     2.0f / 1,
                                     2.0f / 2,
                                     0.0f / 2,
                                     0.0f / 1};
            }
            case 1:
            {
                float denom = 3.0;
                return vector<float>{
                    1 / denom, 3 / denom, 3 / denom, 3 / denom, 4 / denom, 5 / denom, 5 / denom,
                    2 / denom, 2 / denom, 2 / denom, 2 / denom, 0 / denom, 0 / denom, 2 / denom,
                    2 / denom, 2 / denom, 2 / denom, 5 / denom, 5 / denom, 4 / denom, 3 / denom,
                    3 / denom, 3 / denom, 1 / denom, 3 / denom, 4 / denom, 2 / denom, 1 / denom,
                    0 / denom, 2 / denom, 2 / denom, 3 / denom, 1 / denom, 1 / denom, 1 / denom,
                    3 / denom, 3 / denom, 1 / denom, 1 / denom, 1 / denom, 3 / denom, 2 / denom,
                    2 / denom, 0 / denom, 1 / denom, 2 / denom, 4 / denom, 3 / denom};
            }
            }
        }
    }
}

// Interpret Fucntion Test
TEST(nnfusion_ir, avg_pool)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::AvgPool>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::AvgPool::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::AvgPool>(translated);
    EXPECT_TRUE(op != nullptr);

    // Test member function
    // Check fields
    EXPECT_FALSE(op->isTranslated);
    EXPECT_TRUE(op->node != nullptr);
    EXPECT_TRUE(op->args.size() != 0);
    //\todo check tensor
    EXPECT_TRUE(op->out.size() != 0);
    //\todo will check tensor descriptor
    //\todo Check the name generated this pahse
    // Shape input_shape, result_shape, padding_below, padding_above, window_shape,
    //            window_stride
    /*
    print_vector(op->input_shape, "input_shape");
    print_vector(op->result_shape, "result_shape");
    print_vector(op->padding_below, "padding_below");
    print_vector(op->padding_above, "padding_above");
    print_vector(op->window_shape, "window_shape");
    print_vector(op->window_stride, "window_stride");
    */

    EXPECT_TRUE(compare_vector(op->input_shape, Shape{1, 1, 3, 3}));
    EXPECT_TRUE(compare_vector(op->result_shape, Shape{1, 1, 4, 4}));
    EXPECT_TRUE(compare_vector(op->padding_below, Shape{1, 1}));
    EXPECT_TRUE(compare_vector(op->padding_above, Shape{1, 1}));
    EXPECT_TRUE(compare_vector(op->window_shape, Shape{2, 2}));
    EXPECT_TRUE(compare_vector(op->window_stride, Shape{1, 1}));
}