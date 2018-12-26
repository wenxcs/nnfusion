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

            auto graph = tensorflow_import::TensorflowGraph{tensorflow_graph};

            std::vector<std::shared_ptr<Function>> output_functions{graph.get_outputs()};
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

        // void register_operator(const std::string& name,
        //                        std::int64_t version,
        //                        const std::string& domain,
        //                        Operator fn)
        // {
        //     OperatorsBridge::register_operator(name, version, domain, std::move(fn));
        // }

    } // namespace frontend

} // namespace ngraph
