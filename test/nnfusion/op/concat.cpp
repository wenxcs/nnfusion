// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for Concat
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/op/concat.hpp"
#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace inventory
    {
        // We use float by default
        template <>
        shared_ptr<op::Concat> create_object(int option)
        {
            switch (option)
            {
            case 0:
                Shape shape_a{2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                Shape shape_b{2, 3};
                auto B = make_shared<op::Parameter>(element::f32, shape_b);
                Shape shape_c{2, 3};
                auto C = make_shared<op::Parameter>(element::f32, shape_c);
                Shape shape_r{2, 8};
                return make_shared<op::Concat>(NodeVector{A, B, C}, 1);
            }
        }

        template <>
        vector<float> generate_input<op::Concat, float>(int option)
        {
            switch (option)
            {
            case 0:
                return vector<float>{
                    /*a*/ 2,
                    4,
                    8,
                    16,
                    /*b*/ 1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    /*c*/ 2,
                    3,
                    5,
                    7,
                    11,
                    13,
                };
            };
        }

        template <>
        vector<float> generate_output<op::Concat, float>(int option)
        {
            switch (option)
            {
            case 0: return vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13};
            }
        }
    }
}

// Interpret Fucntion Test
TEST(nnfusion_ir, concat)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Concat>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Concat::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Concat>(translated);
    EXPECT_TRUE(op != nullptr);

    // Test member function
    // Check fields
    EXPECT_FALSE(op->isTranslated);
    EXPECT_TRUE(op->node != nullptr);
    EXPECT_TRUE(op->args.size() != 0);
    EXPECT_TRUE(op->out.size() != 0);

    /*
     size_t axis;
            vector<NVShape> input_shapes;
            string dtype;
            Shape output_shape;

    */
    EXPECT_TRUE(op->axis == 1);
    EXPECT_TRUE(op->dtype == "float");
    EXPECT_TRUE(compare_vector(op->input_shapes[0], vector<int>{2, 2}));
    EXPECT_TRUE(compare_vector(op->input_shapes[1], vector<int>{2, 3}));
    EXPECT_TRUE(compare_vector(op->input_shapes[2], vector<int>{2, 3}));
    EXPECT_TRUE(compare_vector(op->output_shape, vector<int>{2, 8}));
}