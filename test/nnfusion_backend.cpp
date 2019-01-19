//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/frontend/tensorflow_import/tensorflow.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

#include "ngraph/file_util.hpp"

using namespace ngraph;

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;
using Model = std::vector<std::shared_ptr<Function>>;

TEST(nnfusion_backend, relu_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_relu_graph.pb"));

    for (auto function : model)
    {
        auto backend = ngraph::runtime::Backend::create("NNFUSION");
        backend->compile(function);
    }
}