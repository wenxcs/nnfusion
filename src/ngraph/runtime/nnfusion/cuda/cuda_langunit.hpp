// Microsoft (c) 2019, Wenxiang
#pragma once
#include "../core/languageunit.hpp"

#define LU_DEFINE(NAME) extern LanguageUnit_p NAME;

namespace nnfusion
{
    namespace cuda
    {
        namespace header
        {
            LU_DEFINE(cuda);
            LU_DEFINE(stdio);
            LU_DEFINE(fstream);
        }

        namespace macro
        {
            LU_DEFINE(NNFUSION_DEBUG);
        }

        namespace declaration
        {
            LU_DEFINE(typedef_int);
            LU_DEFINE(division_by_invariant_multiplication);
            LU_DEFINE(load);
        }
    }
}

#undef LU_DEFINE