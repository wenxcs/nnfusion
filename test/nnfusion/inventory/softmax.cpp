// Microsoft (c) 2019, Yuchao
/**
 * \brief Unit tests for Softmax
 * \author Yuchao Zheng
 */

#include "ngraph/runtime/nnfusion/op/softmax.hpp"
#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace inventory
    {
        // We use float by default
        template <>
        shared_ptr<op::Softmax> create_object(int option)
        {
            switch (option)
            {
            case 0:
            {
                // Parameter
                Shape shape_a{2, 2, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);

                // Softmax
                auto op = make_shared<op::Softmax>(A, AxisVector{0});

                return op;
            }
            case 1:
            {
                // Parameter
                Shape shape_a{2, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);

                // Softmax
                auto op = make_shared<op::Softmax>(A, AxisVector{1});

                return op;
            }
            }
            return nullptr;
        }

        template <>
        vector<float> generate_input<op::Softmax, float>(int option)
        {
            switch (option)
            {
            case 0: { return vector<float>{-10, -20, -30, -40, -50, -60, -1, -2, -3, -4, -5, -6};
            }
            case 1: { return vector<float>{-10, -20, -30, -40, -50, -60};
            }
            }
        }

        template <>
        vector<float> generate_output<op::Softmax, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                auto d0 = expf(-10) + expf(-1);
                auto d1 = expf(-20) + expf(-2);
                auto d2 = expf(-30) + expf(-3);
                auto d3 = expf(-40) + expf(-4);
                auto d4 = expf(-50) + expf(-5);
                auto d5 = expf(-60) + expf(-6);
                return vector<float>{expf(-10) / d0,
                                     expf(-20) / d1,
                                     expf(-30) / d2,
                                     expf(-40) / d3,
                                     expf(-50) / d4,
                                     expf(-60) / d5,
                                     expf(-1) / d0,
                                     expf(-2) / d1,
                                     expf(-3) / d2,
                                     expf(-4) / d3,
                                     expf(-5) / d4,
                                     expf(-6) / d5};
            }
            case 1:
            {
                auto d0 = expf(-10) + expf(-20) + expf(-30);
                auto d1 = expf(-40) + expf(-50) + expf(-60);
                return vector<float>{expf(-10) / d0,
                                     expf(-20) / d0,
                                     expf(-30) / d0,
                                     expf(-40) / d1,
                                     expf(-50) / d1,
                                     expf(-60) / d1};
            }
            }
        }
    }
}

// Interpret Fucntion Test
TEST(nnfusion_ir, softmax_axis_3d)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Softmax>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Softmax::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Softmax>(translated);
    EXPECT_TRUE(op != nullptr);

    // Test member function
    // Check fields
    EXPECT_TRUE(op->node != nullptr);
    EXPECT_TRUE(op->args.size() == 1);
    EXPECT_TRUE(op->out.size() == 1);
}

// Interpret Fucntion Test
TEST(nnfusion_ir, softmax_axis)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Softmax>(1);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Softmax::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Softmax>(translated);
    EXPECT_TRUE(op != nullptr);

    // Test member function
    // Check fields
    EXPECT_TRUE(op->node != nullptr);
    EXPECT_TRUE(op->args.size() == 1);
    EXPECT_TRUE(op->out.size() == 1);
}