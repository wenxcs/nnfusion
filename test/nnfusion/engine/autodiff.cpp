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
#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "nnfusion/engine/external/backend_manager.hpp"
#include "nnfusion/engine/pass/graph/autodiff_pass.hpp"
#include "nnfusion/frontend/onnx_import/onnx.hpp"

using namespace nnfusion;

namespace
{
    using RawInputs = vector<vector<char>>;
    using RawOutputs = vector<vector<char>>;

    template <typename T, typename S = T>
    std::vector<T> convert_from_raw(std::vector<char> src)
    {
        NNFUSION_CHECK(src.size() % sizeof(S) == 0);
        S* raw_data_ptr = (S*)src.data();
        auto src_data_size = src.size() / sizeof(S);
        return vector<T>(raw_data_ptr, raw_data_ptr + src_data_size);
    }

    template <typename T>
    std::vector<char> convert_to_raw(std::vector<T> src)
    {
        auto raw_size = src.size() * sizeof(T);
        char* src_data_ptr = (char*)src.data();
        return vector<char>(src_data_ptr, src_data_ptr + raw_size);
    }

    vector<vector<char>> mixed_type_execute(const shared_ptr<nnfusion::graph::Graph>& graph,
                                            vector<vector<char>> args,
                                            const string& backend_id)
    {
        auto parms_gnodes = graph->get_parameters();

        NNFUSION_CHECK(parms_gnodes.size() == args.size())
            << "number of parameters and arguments don't match";

        auto graph_evaluate = make_shared<nnfusion::profiler::GraphEvaluate>(graph, CUDA_GPU);
        auto res = graph_evaluate->mixed_type_eval(args);

        auto output_gnodes = graph->get_outputs();
        vector<vector<char>> result_vectors;
        ///\todo: we don't have output index yet, so we think it's legal either:
        //1. output_gnode size equal to profiler output size
        //2. only have one profiler output, which contains multiple vector
        if (output_gnodes.size() == res.size())
        {
            for (auto output_gnode : output_gnodes)
            {
                auto gonde_res = res[output_gnode->get_unique_name()];
                // remove this constrain
                NNFUSION_CHECK(gonde_res.size() == 1);
                result_vectors.push_back((gonde_res[0]));
            }
        }
        else if (res.size() == 1)
        {
            result_vectors = res.begin()->second;
            NNFUSION_CHECK(result_vectors.size() == output_gnodes.size());
        }
        else
        {
            NNFUSION_CHECK_FAIL() << "input/output count mismatch";
        }

        return result_vectors;
    }

    void build_backward_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
    {
        FLAGS_fautodiff = true;
        auto ad_pass = nnfusion::pass::graph::AutodiffPass();
        ad_pass.run_on_graph(graph);
    }
}

TEST(nnfusion_pass_autodiff, multiply)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mul.onnx"));

    build_backward_graph(model);

    RawInputs raw_inputs;
    // a
    auto a = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    raw_inputs.emplace_back(convert_to_raw(a));
    // b
    auto b = vector<float>{4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    raw_inputs.emplace_back(convert_to_raw(b));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> a_grad{convert_from_raw<float>(raw_outputs.at(1))};
    vector<float> b_grad{convert_from_raw<float>(raw_outputs.at(2))};

    EXPECT_TRUE(test::all_close_f(out, vector<float>{4, 10, 18, 28, 40, 54}));
    EXPECT_TRUE(test::all_close_f(a_grad, b));
    EXPECT_TRUE(test::all_close_f(b_grad, a));
}