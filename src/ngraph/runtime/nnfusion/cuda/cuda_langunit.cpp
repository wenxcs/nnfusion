// Microsoft (c) 2019, Wenxiang
#include "cuda_langunit.hpp"

using namespace nnfusion::cuda;

#define QUOTE(x) #x
#define STR(x) QUOTE(x)
#define LU_DEFINE(NAME, code)                                                                      \
    LanguageUnit_p NAME = LanguageUnit_p(new LanguageUnit(STR(NAME), code));

// Header
LU_DEFINE(header::cuda, "#include <cuda.h>\n");
LU_DEFINE(header::stdio, "#include <stdio.h>\n");
LU_DEFINE(header::fstream, "#include <fstream>\n");

// Macro
LU_DEFINE(macro::NNFUSION_DEBUG, "#define NNFUSION_DEBUG\n");

// Declaration
LU_DEFINE(declaration::typedef_int,
          "typedef signed char int8_t;\ntypedef signed short int16_t;typedef signed int "
          "int32_t;\ntypedef signed long int int64_t;\ntypedef unsigned char uint8_t;\ntypedef "
          "unsigned short uint16_t;\ntypedef unsigned int uint32_t;\ntypedef unsigned long int "
          "uint64_t;\n");

#undef LU_DEFINE