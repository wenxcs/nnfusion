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
using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

template <typename T, typename T1 = T>
std::vector<std::vector<T1>> execute(const std::shared_ptr<nnfusion::graph::Graph>& graph,
                                     std::vector<std::vector<T>> args,
                                     const std::string& backend_id)
{
    auto parms_gnodes = graph->get_parameters();

    NNFUSION_CHECK(parms_gnodes.size() == args.size())
        << "number of parameters and arguments don't match";

    auto graph_evaluate = make_shared<nnfusion::profiler::GraphEvaluate>(graph, CUDA_GPU);
    auto res = graph_evaluate->eval<T, T1>(args);

    auto output_gnodes = graph->get_outputs();
    NNFUSION_CHECK(output_gnodes.size() == res.size())
        << "number of outputs and results don't match";

    std::vector<std::vector<T1>> result_vectors;
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
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}
