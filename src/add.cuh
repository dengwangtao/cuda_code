#pragma once

#include <cuda_runtime.h>

#include "common.cuh"
#include "cuda_utils.cuh"



template<typename T>
__global__
void add(T* a, T* b, T* c, sz n)
{
    sz i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}