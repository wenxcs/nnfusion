// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for Sum
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/op/sum.hpp"
#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace inventory
    {
        // We use float by default
        template <>
        shared_ptr<op::Sum> create_object(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape_a{3, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                Shape shape_rt{2};
                return make_shared<op::Sum>(A, AxisSet{0});
            }
            case 1:
            {
                Shape shape{2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                return make_shared<op::Sum>(A, AxisSet{0, 1});
            }
            }
        }

        template <>
        vector<float> generate_input<op::Sum, float>(int option)
        {
            switch (option)
            {
            case 0: return vector<float>{1, 2, 3, 4, 5, 6};
            case 1: return vector<float>{1, 2, 3, 4};
            };
        }

        template <>
        vector<float> generate_output<op::Sum, float>(int option)
        {
            switch (option)
            {
            case 0: return vector<float>{9, 12};
            case 1: return vector<float>{10};
            }
        }
    }
}

// Interpret Fucntion Test
TEST(nnfusion_ir, sum)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Sum>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Sum::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Sum>(translated);
    EXPECT_TRUE(op != nullptr);

    // Test member function
    // Check fields
    EXPECT_FALSE(op->isTranslated);
    EXPECT_TRUE(op->node != nullptr);
    EXPECT_TRUE(op->args.size() != 0);
    EXPECT_TRUE(op->out.size() != 0);

    EXPECT_TRUE(compare_vector(op->reduce_axis, vector<int>{0}));
}