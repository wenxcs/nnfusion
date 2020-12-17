//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <fstream>
#include <iostream>

#include "torch/script.h"
#include "torchscript.hpp"
#include "util/graph_convert.hpp"

namespace nnfusion
{
    namespace frontend
    {
        std::shared_ptr<nnfusion::graph::Graph> load_torchscript_model(const std::string& path)
        {
            return load_torchscript_model(path, std::vector<ParamInfo>());
        }

        std::shared_ptr<nnfusion::graph::Graph>
            load_torchscript_model(const std::string& path, const std::vector<ParamInfo>& inputs)
        {
            std::vector<nnfusion::Shape> input_shapes(inputs.size());
            std::vector<nnfusion::element::Type> input_types(inputs.size());
            for (int i = 0; i < inputs.size(); i++)
            {
                input_shapes[i] = inputs[i].shape;
                input_types[i] = inputs[i].type;
            }

            torch::jit::script::Module module = torch::jit::load(path);

            torch::jit::script::Method m = module.get_method("forward");
            std::shared_ptr<torch::jit::Graph> torchscript_graph = m._lowered_graph().first->copy();
            std::vector<at::Tensor> weights = m._lowered_graph().second;

            auto graph_convert = torchscript_import::GraphConvert(
                torchscript_graph, weights, input_shapes, input_types);

            std::shared_ptr<nnfusion::graph::Graph> graph = graph_convert.get_graph();

            return graph;
        }
    } // namespace frontend
} // namespace nnfusion
