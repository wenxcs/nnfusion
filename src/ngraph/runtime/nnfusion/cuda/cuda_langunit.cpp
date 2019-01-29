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
          "typedef signed char int8_t;\ntypedef signed short int16_t;\ntypedef signed int "
          "int32_t;\ntypedef signed long int int64_t;\ntypedef unsigned char uint8_t;\ntypedef "
          "unsigned short uint16_t;\ntypedef unsigned int uint32_t;\ntypedef unsigned long int "
          "uint64_t;\n");
LU_DEFINE(
    declaration::division_by_invariant_multiplication,
    R"(__device__ __forceinline__ int division_by_invariant_multiplication(int value, int magic, int shift)
{
    int result;
    asm("{\n\t"
        ".reg .pred p;\n\t"
        ".reg .u64 res64;\n\t"
        ".reg .u32 lo32, hi32;\n\t"
        "setp.ne.s32 p, %2, 1;\n\t"
        "mul.wide.u32 res64, %1, %2;\n\t"
        "mov.b64 {lo32, hi32}, res64;\n\t"
        "selp.u32 hi32, hi32, %1, p;\n\t"
        "shr.u32 %0, hi32, %3;\n\t"
        "}" : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}
)");

LU_DEFINE(
    declaration::load,
    R"(__device__ __forceinline__ float  load(const float*  __restrict__ in, int i=0, bool b=true)
{
    float v = 0.0f;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ int32_t  load(const int32_t*  __restrict__ in, int i=0, bool b=true)
{
    int32_t v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ int64_t  load(const int64_t*  __restrict__ in, int i=0, bool b=true)
{
    int64_t v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
)");

#undef LU_DEFINE