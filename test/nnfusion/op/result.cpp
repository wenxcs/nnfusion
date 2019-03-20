// Microsoft (c) 2019, Yuchao
/**
 * \brief Unit tests for result
 * \author Yuchao Zheng
 */

#include "ngraph/runtime/nnfusion/op/result.hpp"
#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace inventory
    {
        // We use float by default
        template <>
        shared_ptr<op::Result> create_object(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape_a{2, 3, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                return make_shared<op::Result>(A);
            }
            }
        }

        template <>
        vector<float> generate_input<op::Result, float>(int option)
        {
            switch (option)
            {
            case 0: return vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
            };
        }

        template <>
        vector<float> generate_output<op::Result, float>(int option)
        {
            switch (option)
            {
            case 0: return vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
            }
        }
    }
}

// Interpret Fucntion Test
TEST(nnfusion_ir, result)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Result>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Result::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Result>(translated);
    EXPECT_TRUE(op != nullptr);

    // Test member function
    // Check fields
    EXPECT_FALSE(op->isTranslated);
    EXPECT_TRUE(op->node != nullptr);
    EXPECT_TRUE(op->args.size() != 0);
    EXPECT_TRUE(op->out.size() != 0);

    EXPECT_TRUE(compare_vector(op->out[0].get_shape(), Shape{2, 3, 2}));
    EXPECT_TRUE(compare_vector(op->args[0].get_shape(), Shape{2, 3, 2}));
}