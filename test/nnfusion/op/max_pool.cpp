// Microsoft (c) 2019, Yuchao
/**
 * \brief Unit tests for MaxPool
 * \author Yuchao Zheng
 */

#include "ngraph/runtime/nnfusion/op/max_pool.hpp"
#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace inventory
    {
        // We use float by default
        template <>
        shared_ptr<op::MaxPool> create_object(int option)
        {
            switch (option)
            {
            case 0:
            {
                // Parameter
                Shape shape_a{1, 1, 3, 3};
                Shape window_shape{2, 2};
                auto window_movement_strides = Strides{1, 1};
                Shape padding_below{1, 1};
                Shape padding_above{1, 1};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);

                // MaxPool
                auto op = make_shared<op::MaxPool>(
                    A, window_shape, window_movement_strides, padding_below, padding_above);

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
            }
            case 1:
            {
                Shape shape_a{2, 2, 14};
                Shape shape_r{2, 2, 12};
                Shape window_shape{3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);

                auto op = make_shared<op::MaxPool>(A, window_shape);
                return op;
            }
            }
            return nullptr;
        }

        template <>
        vector<float> generate_input<op::MaxPool, float>(int option)
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
        vector<float> generate_output<op::MaxPool, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                // tensor 1x1x4x4
                return vector<float>{0.0f,
                                     1.0f,
                                     1.0f,
                                     0.0f,
                                     0.0f,
                                     3.0f,
                                     3.0f,
                                     2.0f,
                                     2.0f,
                                     3.0f,
                                     3.0f,
                                     2.0f,
                                     2.0f,
                                     2.0f,
                                     0.0f,
                                     0.0f};
            }
            case 1:
            {
                return vector<float>{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0, 0, 2, 2, 2,
                                     2, 3, 3, 3, 2, 2, 2, 1, 2, 2, 1, 1, 0, 2, 2, 2,
                                     1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2};
            }
            }
        }
    }
}

// Interpret Fucntion Test
TEST(nnfusion_ir, max_pool)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::MaxPool>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::MaxPool::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::MaxPool>(translated);
    EXPECT_TRUE(op != nullptr);

    // Test member function
    // Check fields
    EXPECT_FALSE(op->isTranslated);
    EXPECT_TRUE(op->node != nullptr);
    EXPECT_TRUE(op->args.size() != 0);
    EXPECT_TRUE(op->out.size() != 0);

    EXPECT_TRUE(compare_vector(op->input_shape, Shape{1, 1, 3, 3}));
    EXPECT_TRUE(compare_vector(op->output_shape, Shape{1, 1, 4, 4}));
    EXPECT_TRUE(compare_vector(op->padding_below, Shape{1, 1}));
    EXPECT_TRUE(compare_vector(op->padding_above, Shape{1, 1}));
    EXPECT_TRUE(compare_vector(op->window_shape, Shape{2, 2}));
    EXPECT_TRUE(compare_vector(op->window_stride, Shape{1, 1}));
}