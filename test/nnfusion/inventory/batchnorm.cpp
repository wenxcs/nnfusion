// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for ir::noop
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/op/batchnorm.hpp"
#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace inventory
    {
        // We use float by default
        template <>
        shared_ptr<op::BatchNormInference> create_object(int option)
        {
            switch (option)
            {
            case 0:
            {
                auto input_shape = Shape{2, 2, 2, 1};
                auto input = make_shared<op::Parameter>(element::f32, input_shape);
                auto mean_shape = Shape{2};
                auto mean = make_shared<op::Parameter>(element::f32, mean_shape);
                auto var_shape = Shape{2};
                auto var = make_shared<op::Parameter>(element::f32, var_shape);
                auto gamma_shape = Shape{2};
                auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
                auto beta_shape = Shape{2};
                auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
                double eps = 0.001;
                auto shape_r = Shape{2, 2, 2, 1};
                auto bn = make_shared<op::BatchNormInference>(eps, gamma, beta, input, mean, var);
                // auto f = make_shared<Function>(BN, op::ParameterVector{A});
                return bn;
            }
            }
        }

        template <>
        vector<float> generate_input<op::BatchNormInference, float>(int option)
        {
            switch (option)
            {
            case 0:
                return vector<float>{0.54881352f,
                                     0.71518934f,
                                     0.60276335f,
                                     0.54488319f,
                                     0.42365479f,
                                     0.64589411f,
                                     0.4375872f,
                                     0.89177299f};
                break;
            };
        }

        template <>
        vector<float> generate_output<op::BatchNormInference, float>(int option)
        {
            switch (option)
            {
            case 0:
                return vector<float>{-0.30327f,
                                     1.1561f,
                                     -0.0963782f,
                                     -0.434702f,
                                     -1.4011f,
                                     0.548275f,
                                     -1.06187f,
                                     1.59295f};
                break;
            }
        }

        template <>
        vector<float> generate_param<op::BatchNormInference, float>(int option)
        {
            switch (option)
            {
            case 0:
                return vector<float>{
                    1.0f,
                    1.0f, // gamma
                    0.0f,
                    0.0f, // beta
                    0.583388f,
                    0.619252f, // mean
                    0.0119972f,
                    0.0282681f, // var
                };
                break;
            }
        }
    }
}

// Interpret Fucntion Test
TEST(nnfusion_ir, batchnorm_inference)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::BatchNormInference>();
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::BatchNorm::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto bn = static_pointer_cast<nnfusion::ir::BatchNorm>(translated);
    EXPECT_TRUE(bn != nullptr);

    EXPECT_TRUE(compare_vector(bn->tensor_shape, Shape{2, 2, 2, 1}));
    EXPECT_TRUE(compare_vector(bn->param_shape, Shape{2}));
    EXPECT_TRUE(bn->epsilon == 1e-3);
    EXPECT_TRUE(bn->dtype == "float");
}