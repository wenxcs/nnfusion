// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for ir::noop
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/op/noop.hpp"
#include "../test_util/common.hpp"

// Interpret Fucntion Test
TEST(nnfusion_ir, noop)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::AvgPool>();
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Noop::translate(node);
    EXPECT_TRUE(translated != nullptr);
    EXPECT_POINTER_TYPE(translated, nnfusion::ir::Noop, op);
}