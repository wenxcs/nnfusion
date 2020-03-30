//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <fstream>

#include "onnx.hpp"
#include "util/graph_convert.hpp"

namespace nnfusion
{
    namespace frontend
    {
        std::shared_ptr<nnfusion::graph::Graph> load_onnx_model(std::istream& sin)
        {
            onnx::ModelProto onnx_graph;
            NNFUSION_CHECK(onnx_graph.ParseFromIstream(&sin))
                << "failure parsing data from the stream";

            NNFUSION_LOG(INFO) << "Import ONNX Graph Size: [" << onnx_graph.ByteSizeLong() << "]";
            auto graph_convert = onnx_import::GraphConvert{onnx_graph};

            std::shared_ptr<nnfusion::graph::Graph> graph = graph_convert.get_graph();
            return graph;
        }

        std::shared_ptr<nnfusion::graph::Graph> load_onnx_model(const std::string& path)
        {
            std::ifstream ifs{path, std::ios::in | std::ios::binary};
            NNFUSION_CHECK(ifs.is_open()) << "failure opening file:" + path;
            return load_onnx_model(ifs);
        }
    } // namespace frontend
} // namespace nnfusion
