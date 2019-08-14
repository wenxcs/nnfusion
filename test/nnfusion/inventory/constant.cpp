// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for Constant
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/op/constant.hpp"
#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace inventory
    {
        // We use float by default
        template <>
        shared_ptr<op::Constant> create_object(int option)
        {
            switch (option)
            {
            case 0:
                Shape shape_in{2, 4};
                vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
                auto constant = make_shared<op::Constant>(element::f32, shape_in, values_in);
                return constant;
            }
        }

        template <>
        vector<float> generate_output<op::Constant, float>(int option)
        {
            switch (option)
            {
            case 0: return vector<float>{0, 1, 2, 3, 4, 5, 6, 7};
            }
        }
    }
}

// Interpret Fucntion Test
TEST(nnfusion_ir, Constant)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Constant>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Constant::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Constant>(translated);
    EXPECT_TRUE(op != nullptr);

    // Test member function
    // Check fields
    EXPECT_FALSE(op->isTranslated);
    EXPECT_TRUE(op->node != nullptr);
    EXPECT_TRUE(op->args.size() == 0);
    EXPECT_TRUE(op->out.size() != 0);
}
