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
        std::shared_ptr<nnfusion::graph::Graph>
            load_onnx_model(std::istream& sin,
                            const std::string& model_dir,
                            const std::unordered_map<std::string, size_t>& dim_params)
        {
            onnx::ModelProto onnx_graph;
            NNFUSION_CHECK(onnx_graph.ParseFromIstream(&sin))
                << "failure parsing data from the stream";

            NNFUSION_LOG(INFO) << "Import ONNX Graph Size: [" << onnx_graph.ByteSizeLong() << "]";
            // TODO: this is a hardcode for BERT training
            // std::map<std::string, size_t> dim_map = {
            //     {"batch", 2}, {"sequence", 512}, {"dynamic_prediction_count", 20}};
            auto graph_convert = onnx_import::GraphConvert{onnx_graph, dim_params, model_dir};

            std::shared_ptr<nnfusion::graph::Graph> graph = graph_convert.get_graph();
            return graph;
        }

        std::shared_ptr<nnfusion::graph::Graph>
            load_onnx_model(const std::string& path,
                            const std::unordered_map<std::string, size_t>& dim_params)
        {
            std::ifstream ifs{path, std::ios::in | std::ios::binary};
            NNFUSION_CHECK(ifs.is_open()) << "failure opening file:" + path;
            string model_dir = "";
            auto pos = path.rfind("/");
            if (pos != std::string::npos)
            {
                model_dir = path.substr(0, pos);
            }
            return load_onnx_model(ifs, model_dir, dim_params);
        }
    } // namespace frontend
} // namespace nnfusion
