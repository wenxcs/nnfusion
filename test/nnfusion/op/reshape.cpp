// Microsoft (c) 2019, Yuchao
/**
 * \brief Unit tests for Reshape
 * \author Yuchao Zheng
 */

#include "ngraph/runtime/nnfusion/op/reshape.hpp"
#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace inventory
    {
        // We use float by default
        template <>
        shared_ptr<op::Reshape> create_object(int option)
        {
            switch (option)
            {
            case 0:
            {
                // Parameter
                Shape shape_a{3, 3};
                Shape shape_r{3, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);

                // Reshape
                auto op = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);

                return op;
            }
            case 1:
            {
                // Parameter
                Shape shape_a{2, 3, 4};
                Shape shape_r{2, 4, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);

                // Reshape
                auto op = make_shared<op::Reshape>(A, AxisVector{0, 2, 1}, shape_r);

                return op;
            }
            case 2:
            {
                // Parameter
                Shape shape_a{2, 2, 3, 3, 2, 4};
                Shape shape_r{3, 2, 2, 4, 3, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);

                // Reshape
                auto op = make_shared<op::Reshape>(A, AxisVector{2, 4, 0, 5, 3, 1}, shape_r);

                return op;
            }
            }
            return nullptr;
        }

        template <>
        vector<float> generate_input<op::Reshape, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a_data(3 * 3);
                for (int i = 0; i < 3 * 3; i++)
                    a_data[i] = float(i + 1);
                return a_data;
            }
            case 1:
            {
                vector<float> a_data(2 * 3 * 4);
                for (int i = 0; i < 2 * 3 * 4; i++)
                    a_data[i] = float(i + 1);
                return a_data;
            }
            case 2:
            {
                vector<float> a_data(2 * 2 * 3 * 3 * 2 * 4);
                for (int i = 0; i < 2 * 2 * 3 * 3 * 2 * 4; i++)
                    a_data[i] = float(i + 1);
                return a_data;
            }
            }
        }

        template <>
        vector<float> generate_output<op::Reshape, float>(int option)
        {
            switch (option)
            {
            case 0: { return vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9};
            }
            case 1:
            {
                return vector<float>{1,  5,  9,  2,  6,  10, 3,  7,  11, 4,  8,  12,
                                     13, 17, 21, 14, 18, 22, 15, 19, 23, 16, 20, 24};
            }
            case 2:
            {
                return vector<float>{
                    1.,   73.,  9.,   81.,  17.,  89.,  2.,   74.,  10.,  82.,  18.,  90.,  3.,
                    75.,  11.,  83.,  19.,  91.,  4.,   76.,  12.,  84.,  20.,  92.,  145., 217.,
                    153., 225., 161., 233., 146., 218., 154., 226., 162., 234., 147., 219., 155.,
                    227., 163., 235., 148., 220., 156., 228., 164., 236., 5.,   77.,  13.,  85.,
                    21.,  93.,  6.,   78.,  14.,  86.,  22.,  94.,  7.,   79.,  15.,  87.,  23.,
                    95.,  8.,   80.,  16.,  88.,  24.,  96.,  149., 221., 157., 229., 165., 237.,
                    150., 222., 158., 230., 166., 238., 151., 223., 159., 231., 167., 239., 152.,
                    224., 160., 232., 168., 240., 25.,  97.,  33.,  105., 41.,  113., 26.,  98.,
                    34.,  106., 42.,  114., 27.,  99.,  35.,  107., 43.,  115., 28.,  100., 36.,
                    108., 44.,  116., 169., 241., 177., 249., 185., 257., 170., 242., 178., 250.,
                    186., 258., 171., 243., 179., 251., 187., 259., 172., 244., 180., 252., 188.,
                    260., 29.,  101., 37.,  109., 45.,  117., 30.,  102., 38.,  110., 46.,  118.,
                    31.,  103., 39.,  111., 47.,  119., 32.,  104., 40.,  112., 48.,  120., 173.,
                    245., 181., 253., 189., 261., 174., 246., 182., 254., 190., 262., 175., 247.,
                    183., 255., 191., 263., 176., 248., 184., 256., 192., 264., 49.,  121., 57.,
                    129., 65.,  137., 50.,  122., 58.,  130., 66.,  138., 51.,  123., 59.,  131.,
                    67.,  139., 52.,  124., 60.,  132., 68.,  140., 193., 265., 201., 273., 209.,
                    281., 194., 266., 202., 274., 210., 282., 195., 267., 203., 275., 211., 283.,
                    196., 268., 204., 276., 212., 284., 53.,  125., 61.,  133., 69.,  141., 54.,
                    126., 62.,  134., 70.,  142., 55.,  127., 63.,  135., 71.,  143., 56.,  128.,
                    64.,  136., 72.,  144., 197., 269., 205., 277., 213., 285., 198., 270., 206.,
                    278., 214., 286., 199., 271., 207., 279., 215., 287., 200., 272., 208., 280.,
                    216., 288.};
            }
            }
        }
    }
}

// Interpret Fucntion Test
TEST(nnfusion_ir, reshape_2D)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Reshape>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Reshape::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Reshape>(translated);
    EXPECT_TRUE(op != nullptr);

    // Test member function
    // Check fields
    EXPECT_TRUE(op->node != nullptr);
    EXPECT_TRUE(op->args.size() != 0);
    EXPECT_TRUE(op->out.size() != 0);

    EXPECT_TRUE(op->arg_rank == 2);
    EXPECT_TRUE(compare_vector(op->arg_shape, Shape{3, 3}));
    EXPECT_TRUE(compare_vector(op->input_order, Shape{1, 0}));
    EXPECT_TRUE(compare_vector(op->result_shape, Shape{3, 3}));
}

// Interpret Fucntion Test
TEST(nnfusion_ir, reshape_3D)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Reshape>(1);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Reshape::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Reshape>(translated);
    EXPECT_TRUE(op != nullptr);

    // Test member function
    // Check fields
    EXPECT_TRUE(op->node != nullptr);
    EXPECT_TRUE(op->args.size() != 0);
    EXPECT_TRUE(op->out.size() != 0);

    EXPECT_TRUE(op->arg_rank == 3);
    EXPECT_TRUE(compare_vector(op->arg_shape, Shape{2, 3, 4}));
    EXPECT_TRUE(compare_vector(op->input_order, Shape{0, 2, 1}));
    EXPECT_TRUE(compare_vector(op->result_shape, Shape{2, 4, 3}));
}

// Interpret Fucntion Test
TEST(nnfusion_ir, reshape_D)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Reshape>(2);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Reshape::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Reshape>(translated);
    EXPECT_TRUE(op != nullptr);

    // Test member function
    // Check fields
    EXPECT_TRUE(op->node != nullptr);
    EXPECT_TRUE(op->args.size() != 0);
    EXPECT_TRUE(op->out.size() != 0);

    EXPECT_TRUE(op->arg_rank == 6);
    EXPECT_TRUE(compare_vector(op->arg_shape, Shape{2, 2, 3, 3, 2, 4}));
    EXPECT_TRUE(compare_vector(op->input_order, Shape{2, 4, 0, 5, 3, 1}));
    EXPECT_TRUE(compare_vector(op->result_shape, Shape{3, 2, 2, 4, 3, 2}));
}