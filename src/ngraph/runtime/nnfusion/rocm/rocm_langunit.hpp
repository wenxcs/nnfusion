// Microsoft (c) 2019, Wenxiang
#pragma once
#include "../core/languageunit.hpp"

#define LU_DEFINE(NAME) extern LanguageUnit_p NAME;

namespace nnfusion
{
    namespace rocm
    {
        namespace header
        {
            LU_DEFINE(nnfusion_hip);
        }

        namespace macro
        {
        }

        namespace declaration
        {
            LU_DEFINE(division_by_invariant_multiplication);
            LU_DEFINE(load);
        }

        namespace file
        {
            LU_DEFINE(rocc);
            LU_DEFINE(hipify_rocc);
            LU_DEFINE(cudnn_h);
            LU_DEFINE(cublas_v2_h);
        }
    }
}

#undef LU_DEFINE

#define LU_DEFINE(NAME, name, code)                                                                \
    LanguageUnit_p NAME = LanguageUnit_p(new LanguageUnit(name, code));

#define QUOTE(x) #x
#define STR(x) QUOTE(x)
#define LU_DEFINE_S(NAME, code)                                                                    \
    LanguageUnit_p NAME = LanguageUnit_p(new LanguageUnit(STR(NAME), code));