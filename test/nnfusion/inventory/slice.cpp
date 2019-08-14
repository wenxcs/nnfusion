// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for Slice
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/op/slice.hpp"
#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace inventory
    {
        // We use float by default
        template <>
        shared_ptr<op::Slice> create_object(int option)
        {
            switch (option)
            {
            case 0:
                Shape shape_a{4, 4};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                Shape shape_r{3, 2};
                auto r = make_shared<op::Slice>(A, Coordinate{0, 1}, Coordinate{3, 3});
                return r;
            }
        }

        template <>
        vector<float> generate_input<op::Slice, float>(int option)
        {
            switch (option)
            {
            case 0: return vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            };
        }

        template <>
        vector<float> generate_output<op::Slice, float>(int option)
        {
            switch (option)
            {
            case 0: return vector<float>{2, 3, 6, 7, 10, 11};
            }
        }
    }
}

// Interpret Fucntion Test
TEST(nnfusion_ir, slice)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Slice>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Slice::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Slice>(translated);
    EXPECT_TRUE(op != nullptr);

    // Test member function
    // Check fields
    EXPECT_FALSE(op->isTranslated);
    EXPECT_TRUE(op->node != nullptr);
    EXPECT_TRUE(op->args.size() != 0);
    EXPECT_TRUE(op->out.size() != 0);

    EXPECT_TRUE(compare_vector(op->input_shape, vector<int>{4, 4}));
    EXPECT_TRUE(compare_vector(op->lower_bounds, vector<int>{0, 1}));
    EXPECT_TRUE(compare_vector(op->slice_strides, vector<int>{1, 1}));
    EXPECT_TRUE(compare_vector(op->output_shape, vector<int>{3, 2}));
}