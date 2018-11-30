//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "ngraph/except.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace error
        {
            struct UnknownOperator : ngraph_error
            {
                UnknownOperator(const std::string& name, const std::string& framework)
                    : ngraph_error{(framework.empty() ? "" : framework + ".") + name}
                {
                }
            };

            struct UnknownFramework : ngraph_error
            {
                explicit UnknownFramework(const std::string& framework)
                    : ngraph_error{framework}
                {
                }
            };

            struct file_open : ngraph_error
            {
                explicit file_open(const std::string& path)
                    : ngraph_error{"failure opening file:" + path}
                {
                }
            };

            struct stream_parse : ngraph_error
            {
                explicit stream_parse(std::istream&)
                    : ngraph_error{"failure parsing data from the stream"}
                {
                }
            };
        } // namespace error
    }
}