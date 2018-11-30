//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <string>

#include "util/graph.hpp"

namespace ngraph
{
    namespace frontend
    {
        // Registers TensorFlow custom operator
        // void register_operator(const std::string& name,
        //                        tensorflow_import::Operator fn);

        // Convert on TensorFlow model to a vector of nGraph Functions (input stream)
        std::vector<std::shared_ptr<Function>> load_tensorflow_model(std::istream&);

        // Convert an TensorFlow model to a vector of nGraph Functions
        std::vector<std::shared_ptr<Function>> load_tensorflow_model(const std::string&);

        // // Convert the first output of an TensorFlow model to an nGraph Function (input stream)
        // std::shared_ptr<Function> import_onnx_function(std::istream&);

        // // Convert the first output of an TensorFlow model to an nGraph Function
        // std::shared_ptr<Function> import_onnx_function(const std::string&);

    } // namespace tensorflow_import

} // namespace ngraph
