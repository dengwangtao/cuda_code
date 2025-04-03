#pragma once

#ifndef COMMON_CUH
#define COMMON_CUH

#include <random>
#include <driver_types.h>

using u32 = unsigned int;
using s32 = int;
using u64 = unsigned long long;
using s64 = long long;
using f32 = float;
using f64 = double;

using sz = size_t;






#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code!= cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: err<%s> %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}








namespace CommonUtils
{



template<typename T>
T random(const T& min, const T& max)
{
    // C++随机数
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<f64> dist(min, max);
    return static_cast<T>(dist(rng));
}



} // namespace CommonUtils

#endif // COMMON_CUH