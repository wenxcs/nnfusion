// Microsoft (c) 2020, NNFusion Team

#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/languageunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        class AntaresKEImp
        {
        public:
            using Pointer = shared_ptr<AntaresKEImp>;
            AntaresKEImp() {}
            std::string autogen(const std::string& expr, bool antares_quick_codegen = false);
            static std::unordered_map<std::string, std::string> code_cache;
        };
    } // namespace kernels
} // namespace nnfusion
