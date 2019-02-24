// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for Broadcast
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/op/broadcast.hpp"
#include "../test_util/common.hpp"
#include "ngraph/runtime/nnfusion/op/result.hpp"

namespace nnfusion
{
    namespace inventory
    {
        // We use float by default
        template <>
        shared_ptr<op::Broadcast> create_object(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape_a{};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                Shape shape_r{4};
                return make_shared<op::Broadcast>(A, shape_r, AxisSet{0});
            }
            break;
            case 1:
            {
                Shape shape_a{4};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                Shape shape_r{4};
                return make_shared<op::Broadcast>(A, shape_r, AxisSet());
            }
            break;
            }
        }

        template <>
        vector<float> generate_input<op::Broadcast, float>(int option)
        {
            switch (option)
            {
            case 0: return vector<float>{6}; break;
            };
        }

        template <>
        vector<float> generate_output<op::Broadcast, float>(int option)
        {
            switch (option)
            {
            case 0: return vector<float>{6, 6, 6, 6}; break;
            }
        }
    }
}

// Interpret Fucntion Test
TEST(nnfusion_ir, broadcast)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Broadcast>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Broadcast::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Broadcast>(translated);
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
    print_vector(op->arg_shape, "arg_shape");
    print_vector(op->result_shape, "result_shape");
    print_set(op->axes, "axes");
    print_vector(op->strides, "strides");
    print_vector(op->stride_magic, "stride_magic");
    print_vector(op->stride_shift, "stride_shift");
    print_vector(op->reduced_shape, "reduced_shape");
    print_vector(op->reduced_strides, "reduced_strides");
    */

    EXPECT_TRUE(op->result_shape == Shape{4});
    EXPECT_TRUE(op->strides == ngraph::NVShape{1});
    EXPECT_TRUE(op->stride_magic == vector<int>{1});
    EXPECT_TRUE(op->stride_shift == vector<int>{0});
    EXPECT_TRUE(op->reduced_shape == ngraph::NVShape{1});
    EXPECT_TRUE(op->reduced_strides == ngraph::NVShape{0});
    EXPECT_TRUE(op->axes == AxisSet{0});
}

TEST(nnfusion_ir, broadcast_same_shape)
{
    auto node = nnfusion::inventory::create_object<op::Broadcast>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Broadcast::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Result>(translated);
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

    //\todo check result op
}