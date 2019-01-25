// Microsoft (c) 2019, Wenxiang
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_langunit.hpp"

#define QUOTE(x) #x
#define STR(x) QUOTE(x)
#define LU_DEFINE(NAME, code)                                                                      \
    extern shared_ptr<LanguageUnit> NAME =                                                         \
        shared_ptr<LanguageUnit>(new LanguageUnit(STR(NAME), code));

// Header
LU_DEFINE(header::cuda, "#include <cuda.h>\n");
LU_DEFINE(header::stdio, "#include <stdio.h>\n");

// Macro
LU_DEFINE(macro::NNFUSION_DEBUG, "#define NNFUSION_DEBUG\n");

// Declaration
LU_DEFINE(declaration::typedef_int,
          "typedef signed char int8_t;\ntypedef signed short int16_t;typedef signed int "
          "int32_t;\ntypedef signed long int int64_t;\ntypedef unsigned char uint8_t;\ntypedef "
          "unsigned short uint16_t;\ntypedef unsigned int uint32_t;\ntypedef unsigned long int "
          "uint64_t;\n");

#undef LU_DEFINE