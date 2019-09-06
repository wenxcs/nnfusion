//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "ngraph/except.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace error
        {
            struct UnknownOperator : ngraph::ngraph_error
            {
                UnknownOperator(const std::string& name, const std::string& framework)
                    : ngraph_error{(framework.empty() ? "" : framework + ".") + name}
                {
                }
            };

            struct UnknownFramework : ngraph::ngraph_error
            {
                explicit UnknownFramework(const std::string& framework)
                    : ngraph_error{framework}
                {
                }
            };

            struct file_open : ngraph::ngraph_error
            {
                explicit file_open(const std::string& path)
                    : ngraph_error{"failure opening file:" + path}
                {
                }
            };

            struct stream_parse : ngraph::ngraph_error
            {
                explicit stream_parse(std::istream&)
                    : ngraph_error{"failure parsing data from the stream"}
                {
                }
            };
        } // namespace error
    }
}