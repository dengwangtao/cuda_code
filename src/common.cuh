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
    T randomImpl(const T& min, const T& max, std::true_type)
    {
        // C++随机数
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_int_distribution<T> dist(min, max);
        return dist(rng);
    }

    template<typename T>
    T randomImpl(const T& min, const T& max, std::false_type)
    {
        // C++随机数
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_real_distribution<T> dist(min, max);
        return dist(rng);
    }

    template<typename T>
    T random(const T& min, const T& max)
    {
        return randomImpl<T>(min, max, std::is_integral<T>{});
    }


} // namespace CommonUtils

#endif // COMMON_CUH