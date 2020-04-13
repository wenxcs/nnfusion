//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "../test_util/common.hpp"
#include "gtest/gtest.h"
#include "nnfusion/engine/external/backend_manager.hpp"
#include "nnfusion/frontend/onnx_import/onnx.hpp"

using namespace nnfusion;
using Inputs = vector<vector<float>>;
using Outputs = vector<vector<float>>;

template <typename T, typename T1 = T>
vector<vector<T1>> execute(const shared_ptr<nnfusion::graph::Graph>& graph,
                           vector<vector<T>> args,
                           const string& backend_id)
{
    auto parms_gnodes = graph->get_parameters();

    NNFUSION_CHECK(parms_gnodes.size() == args.size())
        << "number of parameters and arguments don't match";

    auto graph_evaluate = make_shared<nnfusion::profiler::GraphEvaluate>(graph, CUDA_GPU);
    auto res = graph_evaluate->eval<T, T1>(args);

    auto output_gnodes = graph->get_outputs();
    NNFUSION_CHECK(output_gnodes.size() == res.size())
        << "number of outputs and results don't match";

    vector<vector<T1>> result_vectors;
    for (auto output_gnode : output_gnodes)
    {
        auto gonde_res = res[output_gnode->get_unique_name()];
        NNFUSION_CHECK(gonde_res.size() == 1);
        result_vectors.push_back((gonde_res[0]));
    }

    return result_vectors;
}

TEST(nnfusion_onnx_import, abs_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/abs.onnx"));

    Inputs inputs{{1, -3.4, 0}};
    Outputs expected_outputs{{1, 3.4, 0}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, add_abc_initializers_op)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc_initializers.onnx"));
    // A + B + C, A is in initializer {1,2,3,4} B is constant {1,2,3,4}
    Inputs inputs{{1, 2, 3, 4}};
    Outputs expected_outputs{{3, 6, 9, 12}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, add_abc_op)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    Inputs inputs{{1}, {2}, {3}};
    Outputs expected_outputs{{6}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, add_bcast_op)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_bcast.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>(
                            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                            .get_vector());

    inputs.emplace_back(test::NDArray<float, 1>({1, 2, 3, 4, 5}).get_vector());

    Outputs expected_outputs{
        test::NDArray<float, 4>(
            {{{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
              {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
              {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}}})
            .get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, addmul_abc_op)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/addmul_abc.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{9, 10}, {11, 12}}}).get_vector());
    inputs.emplace_back(test::NDArray<float, 3>({{{5, 6}, {7, 8}}}).get_vector());
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2}, {3, 4}}}).get_vector());

    Outputs expected_outputs{test::NDArray<float, 3>({{{46, 62}, {80, 100}}}).get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, acos_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/acos.onnx"));

    Inputs inputs{{0, 1, -0.5}};
    Outputs expected_outputs{{1.5708, 0.0, 2.0944}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, and_op)
{
    // cast op is used
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/and.onnx"));

    vector<vector<int64_t>> inputs{{1, 0, 1, 0}, {1, 1, 0, 0}};
    vector<vector<int64_t>> expected_outputs{{1, 0, 0, 0}};

    vector<vector<int64_t>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, asin_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/asin.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{-0.5236, 0.0000, 1.5708}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}
/* no kernel implemented for argmax and argmin
TEST(nnfusion_onnx_import, argmax_int32_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/argmax_int32.onnx"));

    vector<vector<int32_t>> inputs{
        vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};

    vector<vector<int64_t>> expected_outputs{
        vector<int64_t>{1, 1, 1, 1, 1, 1}};

    vector<vector<int64_t>> outputs{execute<int32_t, int64_t>(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, argmin_int32_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_int32.onnx"));

    vector<vector<int32_t>> inputs{
        vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};

    vector<vector<int64_t>> expected_outputs{vector<int64_t>{0, 0, 0, 0}};

    vector<vector<int64_t>> outputs{execute<int32_t, int64_t>(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, argmin_no_keepdims)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_no_keepdims.onnx"));

    Inputs inputs{{2, 1, 3, 10}};
    Outputs expected_outputs{{1, 0}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}
*/
TEST(nnfusion_onnx_import, atan_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/atan.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{-0.4636, 0.0000, 0.7854}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, average_pool_2d_op)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/average_pool_2d.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(test::NDArray<float, 4>({{{{0.f, 1.f, 2.f, 3.f},
                                                {4.f, 5.f, 6.f, 7.f},
                                                {8.f, 9.f, 10.f, 11.f},
                                                {12.f, 13.f, 14.f, 15.f}}}})
                         .get_vector());

    // (1, 1, 2, 2)
    Outputs expected_outputs{
        test::NDArray<float, 4>({{{{2.5f, 4.5f}, {10.5f, 12.5f}}}}).get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, average_pool_2d_pads_op)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/average_pool_2d_pads.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(test::NDArray<float, 4>({{{{0.f, 1.f, 2.f, 3.f},
                                                {4.f, 5.f, 6.f, 7.f},
                                                {8.f, 9.f, 10.f, 11.f},
                                                {12.f, 13.f, 14.f, 15.f}}}})
                         .get_vector());

    // (1, 1, 3, 3)
    Outputs expected_outputs{
        test::NDArray<float, 4>({{{{0.f, 1.5f, 3.f}, {6.f, 7.5f, 9.f}, {12.f, 13.5f, 15.f}}}})
            .get_vector()};
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, batch_norm_op)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default.onnx"));

    Inputs inputs;
    inputs.push_back({-1.f, 0.f, 1.f, 2.f, 3.f, 4.f}); // data {1, 2, 1, 3}
    inputs.push_back({1.f, 1.5f});                     // scale
    inputs.push_back({0.f, 1.f});                      // bias
    inputs.push_back({0.f, 3.f});                      // mean
    inputs.push_back({1.f, 1.5f});                     // var

    Outputs expected_outputs{{-0.999995f, 0.f, 0.999995f, -0.22474074f, 1.f, 2.2247407f}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, ceil_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/ceil.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{-0., 0., 1.}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, concat_op)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/concat.onnx"));

    Inputs inputs;

    inputs.emplace_back(test::NDArray<float, 1>({1, 2}).get_vector());
    inputs.emplace_back(test::NDArray<float, 1>({3, 4}).get_vector());

    Outputs expected_outputs{test::NDArray<float, 1>({1, 2, 3, 4}).get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, cos_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/cos.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{0.8776, 1.0000, 0.5403}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, div_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/div.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 3>({{{1, 4, 12}}}).get_vector());

    Outputs expected_outputs{test::NDArray<float, 3>({{{1, 0.5, 0.25}}}).get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, exp_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/exp.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{0.6065, 1.0000, 2.7183}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, floor_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/floor.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{-1, 0, 1}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, log_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/log.onnx"));

    Inputs inputs{{0.5, 1, 2}};
    Outputs expected_outputs{{-0.6931, 0.0000, 0.6931}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, max_pool_2d_pads_op)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/max_pool_2d_pads.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(test::NDArray<float, 4>({{{{0.f, 1.f, 2.f, 3.f},
                                                {4.f, 5.f, 6.f, 7.f},
                                                {8.f, 9.f, 10.f, 11.f},
                                                {12.f, 13.f, 14.f, 15.f}}}})
                         .get_vector());

    // (1, 1, 3, 3)
    Outputs expected_outputs{
        test::NDArray<float, 4>({{{{0.f, 2.f, 3.f}, {8.f, 10.f, 11.f}, {12.f, 14.f, 15.f}}}})
            .get_vector()};
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, pow_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/pow.onnx"));

    vector<vector<int64_t>> inputs{{1, 2, 4}, {3, 1, 2}};
    vector<vector<int64_t>> expected_outputs{{1, 2, 16}};

    vector<vector<int64_t>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, relu_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/relu.onnx"));

    Inputs inputs{{-0.5, 0, 1, -1.2, 2.4, -5}};
    Outputs expected_outputs{{0.0000, 0.0000, 1.0000, 0.0000, 2.4000, 0.0000}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, sigmoid_op)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sigmoid.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{0.3775, 0.5000, 0.7311}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, sin_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sin.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{-0.4794, 0.0000, 0.8415}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, sqrt_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sqrt.onnx"));

    Inputs inputs{{0.0, 1.0, 4.0, 5.0}};
    Outputs expected_outputs{{0.0000, 1.0000, 2.0000, 2.2361}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, sub_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sub.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 3>({{{4, 5, 7}}}).get_vector());

    Outputs expected_outputs{test::NDArray<float, 3>({{{-3, -3, -4}}}).get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, tan_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/tan.onnx"));

    Inputs inputs{{-1, 0, 1}};
    Outputs expected_outputs{{-1.5574, 0.0000, 1.5574}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, tanh_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/tanh.onnx"));

    Inputs inputs{{-1, 0, 1}};
    Outputs expected_outputs{{-0.7616, 0.0000, 0.7616}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}