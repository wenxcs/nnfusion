// Microsoft (c) 2019, Wenxiang
#include "rocm_langunit.hpp"

using namespace nnfusion;

LU_DEFINE_S(
    rocm::declaration::division_by_invariant_multiplication,
    R"(__device__ __forceinline__ int division_by_invariant_multiplication(int value, int magic, int shift)
{
        long long res64 = ((long long)value) * ((long long)magic);
        int lo32 = res64 & (-1);
        int hi32 = res64 >> 32;
        if(magic == 1)
                hi32 = value;
        int result = hi32 >> shift;
        return result;
}
)");

LU_DEFINE_S(rocm::header::nnfusion_hip, "#include \"nnfusion_hip.h\"\n");

LU_DEFINE_S(
    rocm::declaration::load,
    R"(__device__ __forceinline__ float  load(const float*  __restrict__ in, int i=0, bool b=true)
{
    float v = 0.0f;
    if (b)
    {
        v = in[i];
    }
    return v;
}
__device__ __forceinline__ int32_t  load(const int32_t*  __restrict__ in, int i=0, bool b=true)
{
    int32_t v = 0;
    if (b)
    {
        v = in[i];
    }
    return v;
}
__device__ __forceinline__ int64_t  load(const int64_t*  __restrict__ in, int i=0, bool b=true)
{
    int64_t v = 0;
    if (b)
    {
        v = in[i];
    }
    return v;
}
)");