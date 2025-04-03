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

template<typename T>
__global__
void add(T* a, T* b, T* c, sz row, sz col) // for 2D arrays
{
#if 1
    sz ix = blockIdx.x * blockDim.x + threadIdx.x;
    sz iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < col && iy < row)
    {
        sz i = iy * col + ix;
        c[i] = a[i] + b[i];
    }
#else
    sz block_i = blockIdx.x + blockIdx.y * gridDim.x;
    sz thread_i = threadIdx.x + threadIdx.y * blockDim.x;
    sz i = block_i * (blockDim.x * blockDim.y * blockDim.z) + thread_i;
    if (i < row * col)
    {
        c[i] = a[i] + b[i];
    }
#endif
}