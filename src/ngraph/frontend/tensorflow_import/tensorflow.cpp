//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <fstream>

#include "ngraph/except.hpp"
#include "tensorflow.hpp"

namespace ngraph
{
    namespace frontend
    {
        std::vector<std::shared_ptr<Function>> load_tensorflow_model(std::istream& sin)
        {
            tensorflow::GraphDef tensorflow_graph;
            if (!tensorflow_graph.ParseFromIstream(&sin))
            {
                throw error::stream_parse{sin};
            }
            else
            {
                std::cerr << "Import Tensorflow Graph Size: [" << tensorflow_graph.ByteSizeLong()
                          << "]" << std::endl;
            }

            auto graph_convert = tensorflow_import::GraphConvert{tensorflow_graph};

            std::vector<std::shared_ptr<Function>> output_functions = graph_convert.get_funcs();
            return output_functions;
        }

        std::vector<std::shared_ptr<Function>> load_tensorflow_model(const std::string& path)
        {
            std::ifstream ifs{path, std::ios::in | std::ios::binary};
            if (!ifs.is_open())
            {
                throw frontend::error::file_open{path};
            }
            return load_tensorflow_model(ifs);
        }

        std::shared_ptr<nnfusion::graph::Graph> load_tensorflow_model_as_graph(std::istream& sin)
        {
            tensorflow::GraphDef tensorflow_graph;
            if (!tensorflow_graph.ParseFromIstream(&sin))
            {
                throw error::stream_parse{sin};
            }
            else
            {
                std::cerr << "Import Tensorflow Graph Size: [" << tensorflow_graph.ByteSizeLong()
                          << "]" << std::endl;
            }

            auto graph_convert = tensorflow_import::GraphConvert{tensorflow_graph};

            std::shared_ptr<nnfusion::graph::Graph> graph = graph_convert.get_graph();
            return graph;
        }

        std::shared_ptr<nnfusion::graph::Graph>
            load_tensorflow_model_as_graph(const std::string& path)
        {
            std::ifstream ifs{path, std::ios::in | std::ios::binary};
            if (!ifs.is_open())
            {
                throw frontend::error::file_open{path};
            }
            return load_tensorflow_model_as_graph(ifs);
        }

        // void register_operator(const std::string& name,
        //                        std::int64_t version,
        //                        const std::string& domain,
        //                        Operator fn)
        // {
        //     OperatorsBridge::register_operator(name, version, domain, std::move(fn));
        // }

    } // namespace frontend

} // namespace ngraph
