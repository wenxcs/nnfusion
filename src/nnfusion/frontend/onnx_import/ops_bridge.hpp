//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>

#include "onnx_base.hpp"
namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            class OperatorsBridge
            {
            public:
                OperatorsBridge(const OperatorsBridge&) = delete;
                OperatorsBridge& operator=(const OperatorsBridge&) = delete;
                OperatorsBridge(OperatorsBridge&&) = delete;
                OperatorsBridge& operator=(OperatorsBridge&&) = delete;

                static ConvertFuncMap get_convert_func_map(std::int64_t version,
                                                           const std::string& domain)
                {
                    return instance()._get_convert_func_map(version, domain);
                }

                static void register_operator(const std::string& name,
                                              std::int64_t version,
                                              const std::string& domain,
                                              ConvertFunc fn)
                {
                    instance()._register_operator(name, version, domain, std::move(fn));
                }

            private:
                std::unordered_map<
                    std::string,
                    std::unordered_map<std::string, std::map<std::int64_t, ConvertFunc>>>
                    m_map;

                OperatorsBridge();

                static OperatorsBridge& instance()
                {
                    static OperatorsBridge instance;
                    return instance;
                }

                void _register_operator(const std::string& name,
                                        std::int64_t version,
                                        const std::string& domain,
                                        ConvertFunc fn);
                ConvertFuncMap _get_convert_func_map(std::int64_t version,
                                                     const std::string& domain);
            };

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
