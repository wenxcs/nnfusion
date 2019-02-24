// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for ir::anyop
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/core/interpreter.hpp"
#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace inventory
    {
        // Create a simple graph(Function) for test
        template <>
        shared_ptr<ngraph::Function> create_object(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape_a{2, 1, 3, 5};
                Shape shape_b{2, 1, 2, 2};
                Shape shape_r{2, 2, 1, 2};
                // To make it const
                vector<float> filter{0.67187500f,
                                     0.54687500f,
                                     -0.56250000f,
                                     -0.35937500f,
                                     -0.09375000f,
                                     0.54687500f,
                                     -0.54687500f,
                                     0.89062500f};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto B = make_shared<op::Parameter>(element::f32, shape_b);
                return make_shared<Function>(
                    make_shared<op::Convolution>(A,
                                                 B,
                                                 Strides{2, 2},        // move_strides
                                                 Strides{1, 1},        // filter_dilation
                                                 CoordinateDiff{0, 0}, // below_pads
                                                 CoordinateDiff{0, 0}, // above_pads
                                                 Strides{1, 1}),       // data_dilation
                    op::ParameterVector{A, B});
            }
            }
        }

        template <>
        vector<float> generate_input<ngraph::Function, float>(int option)
        {
            switch (option)
            {
            case 0:
                return vector<float>{
                    0.67187500f,  0.54687500f,  -0.56250000f, -0.35937500f, -0.09375000f,
                    0.54687500f,  -0.54687500f, 0.89062500f,  0.82812500f,  -0.54687500f,
                    1.00000000f,  -0.07812500f, -0.89062500f, 0.40625000f,  -0.35937500f,
                    0.54687500f,  0.60937500f,  0.59375000f,  0.09375000f,  -0.21875000f,
                    0.76562500f,  0.40625000f,  -0.73437500f, -0.95312500f, -0.50000000f,
                    -0.29687500f, 0.76562500f,  -0.26562500f, -0.50000000f, 0.53125000f};
                break;
            };
        }

        template <>
        vector<float> generate_output<ngraph::Function, float>(int option)
        {
            switch (option)
            {
            case 0:
                return vector<float>{0.63940430f,
                                     -1.37304688f,
                                     -0.55004883f,
                                     0.10668945f,
                                     0.12402344f,
                                     1.20581055f,
                                     0.22509766f,
                                     -0.45166016f};
                break;
            }
        }
    }
}

TEST(nnfusion_core, interpreter_constructor)
{
    auto graph = nnfusion::inventory::create_object<ngraph::Function>();

    // Test  1
    auto obj_0 = shared_ptr<nnfusion::FunctionTranslator>();
    // Test 2
    auto obj_1_pass = shared_ptr<vector<shared_ptr<IFunctionTranslatorPass>>>();
    auto obj_1_ctx = shared_ptr<FunctionTranslatorContext>();
    /*
    auto obj_1 = shared_ptr<nnfusion::FunctionTranslator>(obj_1_pass, obj_1_ctx);
    EXPECT_TRUE(obj_1 != nullptr);
    EXPECT_TRUE(obj_1.m_trans_ctx == obj_1_ctx);
    EXPECT_TRUE(obj_1.m_passes = obj_1_pass);
    */
}