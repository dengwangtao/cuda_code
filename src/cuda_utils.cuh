#pragma once
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include "common.cu"

#include <iostream>


template<typename T>
class CudaMemRAII
{
public:
    CudaMemRAII(T* ptr, size_t size) : m_ptr(ptr), m_size(size) {}
    ~CudaMemRAII()
    {
        if (! m_ptr)
        {
            return;
        }
        CUDA_CHECK(
            cudaFree(m_ptr)
        );
    }
    // 删除拷贝构造函数和赋值运算符
    CudaMemRAII(const CudaMemRAII&) = delete;
    CudaMemRAII& operator=(const CudaMemRAII&) = delete;
    // 移动构造函数和移动赋值运算符 
    CudaMemRAII(CudaMemRAII&& other) : m_ptr(other.m_ptr), m_size(other.m_size)
    {
        other.m_ptr = nullptr;
        other.m_size = 0;
    }
    CudaMemRAII& operator=(CudaMemRAII&& other)
    {
        if (this!= &other)
        {
            m_ptr = other.m_ptr;
            m_size = other.m_size;
            other.m_ptr = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    T* get() const { return m_ptr; }
    T* get() {return m_ptr; }

    size_t size() const
    {
        return m_size;
    }

    size_t bsize() const
    {
        return m_size * sizeof(T);
    }

    void CopyFromHost(const T* host_ptr, size_t size)
    {
        if (size > m_size)
        {
            std::cout << size << " " << m_size << std::endl;
            throw std::runtime_error("CudaMemRAII::CopyFromHost: size > m_size");
        }
        CUDA_CHECK(
            cudaMemcpy(m_ptr, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice)
        );
    }

    void CopyToHost(T* host_ptr, size_t size)
    {
        if (size > m_size)
        {
            throw std::runtime_error("CudaMemRAII::CopyToHost: size > m_size");
        }
        CUDA_CHECK(
            cudaMemcpy(host_ptr, m_ptr, size * sizeof(T), cudaMemcpyDeviceToHost)
        );
    }

    static CudaMemRAII<T> Alloc(size_t size)
    {
        // 必须是 trivially copyable
        static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");

        T* ptr = nullptr;
        CUDA_CHECK(
            cudaMalloc(&ptr, size * sizeof(T))
        );
        return CudaMemRAII<T>(ptr, size);
    }
    
    

private:
    T* m_ptr;
    size_t m_size;
};

#endif // CUDA_UTILS_CUH