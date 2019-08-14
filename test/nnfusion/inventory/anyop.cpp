// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for ir::anyop
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/op/anyop.hpp"
#include "../test_util/common.hpp"

// Interpret Fucntion Test
TEST(nnfusion_ir, anyop)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::AvgPool>();
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Anyop::translate(node);
    EXPECT_TRUE(translated != nullptr);
    EXPECT_POINTER_TYPE(translated, nnfusion::ir::Anyop, op);
    // Member value check
    EXPECT_TRUE(op->dtypes.size() != 0);
    EXPECT_TRUE(op->dtypes[0] == "float");
    EXPECT_TRUE(op->dtypes[1] == "float");

    EXPECT_TRUE(op->dsizes.size() != 0);
    EXPECT_TRUE(op->dsizes[0] == "9");
    EXPECT_TRUE(op->dsizes[1] == "16");
}