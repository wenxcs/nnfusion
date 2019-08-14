// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for Convolution
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/op/convolution.hpp"
#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace inventory
    {
        // We use float by default
        template <>
        shared_ptr<op::Convolution> create_object(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape_a{1, 1, 3, 5};
                Shape shape_b{2, 1, 2, 2};
                Shape shape_r{1, 2, 2, 4};

                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto B = make_shared<op::Parameter>(element::f32, shape_b);
                return make_shared<op::Convolution>(A,
                                                    B,
                                                    Strides{1, 1},        // move_strides
                                                    Strides{1, 1},        // filter_dilation
                                                    CoordinateDiff{0, 0}, // below_pads
                                                    CoordinateDiff{0, 0}, // above_pads
                                                    Strides{1, 1});       // data_dilation
            }
            break;
            }
        }

        template <>
        vector<float> generate_input<op::Convolution, float>(int option)
        {
            switch (option)
            {
            case 0:
                return vector<float>{0.67187500f,
                                     0.54687500f,
                                     -0.56250000f,
                                     -0.35937500f,
                                     -0.09375000f,
                                     0.54687500f,
                                     -0.54687500f,
                                     0.89062500f,
                                     0.82812500f,
                                     -0.54687500f,
                                     1.00000000f,
                                     -0.07812500f,
                                     -0.89062500f,
                                     0.40625000f,
                                     -0.35937500f};
                break;
            };
        }

        template <>
        vector<float> generate_output<op::Convolution, float>(int option)
        {
            switch (option)
            {
            case 0:
                return vector<float>{0.63940430f,
                                     0.04736328f,
                                     -1.37304688f,
                                     -0.56201172f,
                                     -0.46606445f,
                                     0.48364258f,
                                     1.40625000f,
                                     0.15795898f,
                                     -0.55004883f,
                                     0.73339844f,
                                     0.10668945f,
                                     -0.95751953f,
                                     -0.96679688f,
                                     -0.21215820f,
                                     1.21826172f,
                                     -0.91894531f};
                break;
            }
        }

        template <>
        vector<float> generate_param<op::Convolution, float>(int option)
        {
            switch (option)
            {
            case 0:
                return vector<float>{0.67187500f,
                                     0.54687500f,
                                     -0.56250000f,
                                     -0.35937500f,
                                     -0.09375000f,
                                     0.54687500f,
                                     -0.54687500f,
                                     0.89062500f};
                break;
            }
        }
    }
}

// Interpret Fucntion Test
TEST(nnfusion_ir, Convolution)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Convolution>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Convolution::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Convolution>(translated);
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

    /*
    print_vector(op->input_shape, "input_shape");
    print_vector(op->filter_shape, "filter_shape");
    print_vector(op->output_shape, "output_shape");
    print_vector(op->window_dilation_strides, "window_dilation_strides");
    print_vector(op->window_movement_strides, "window_movement_strides");
    print_vector(op->data_dilation_strides, "data_dilation_strides");
    print_vector(op->padding_below_diff, "padding_below_diff");
    print_vector(op->padding_above_diff, "padding_above_diff");
    */

    EXPECT_TRUE(compare_vector(op->input_shape, Shape{1, 1, 3, 5}));
    EXPECT_TRUE(compare_vector(op->filter_shape, Shape{2, 1, 2, 2}));
    EXPECT_TRUE(compare_vector(op->output_shape, Shape{1, 2, 2, 4}));
    EXPECT_TRUE(compare_vector(op->window_dilation_strides, Shape{1, 1}));
    EXPECT_TRUE(compare_vector(op->window_movement_strides, Shape{1, 1}));
    EXPECT_TRUE(compare_vector(op->data_dilation_strides, Shape{1, 1}));
    EXPECT_TRUE(compare_vector(op->padding_above_diff, Shape{0, 0}));
    EXPECT_TRUE(compare_vector(op->padding_above_diff, Shape{0, 0}));
}