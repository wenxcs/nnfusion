// Microsoft (c) 2019, NNFusion Team

#include "common_langunit.hpp"

using namespace nnfusion::kernels;

// Header
LU_DEFINE(header::stdio, "#include <stdio.h>\n");
LU_DEFINE(header::cmath, "#include <cmath>\n");
// LU_DEFINE(header::algorithm, "#include <algorithm>\n");
LU_DEFINE(header::fstream, "#include <fstream>\n");
LU_DEFINE(header::stdexcept, "#include <stdexcept>\n");
LU_DEFINE(header::sstream, "#include <sstream>\n");
LU_DEFINE(header::assert, "#include <assert.h>\n");

// Macro
LU_DEFINE(macro::NNFUSION_DEBUG, "#define NNFUSION_DEBUG\n");
LU_DEFINE(macro::MIN, "#define MIN(a,b) ((a)>(b)?(b):(a))\n")

// Declaration
LU_DEFINE(declaration::typedef_int,
          "typedef signed char int8_t;\ntypedef signed short int16_t;\ntypedef signed int "
          "int32_t;\ntypedef signed long int int64_t;\ntypedef unsigned char uint8_t;\ntypedef "
          "unsigned short uint16_t;\ntypedef unsigned int uint32_t;\ntypedef unsigned long int "
          "uint64_t;\n");