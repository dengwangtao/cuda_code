#include <iostream>
#include <vector>
#include "common.cuh"
#include "add.cuh"
#include "cuda_utils.cuh"
#include <iomanip>


void add_demo()
{
    using DemoType = double;
    sz n = 10000;

    std::vector<DemoType> vec1;
    std::vector<DemoType> vec2;
    std::vector<DemoType> result(n);

    for (sz i = 0; i < n; i++)
    {
        vec1.push_back(CommonUtils::random<DemoType>(0, 100));
        vec2.push_back(CommonUtils::random<DemoType>(0, 100));
    }

    // cuda add

    {
        auto d_vec1 = CudaMemRAII<DemoType>::Alloc(n);
        auto d_vec2 = CudaMemRAII<DemoType>::Alloc(n);

        d_vec1.CopyFromHost(vec1.data(), n);
        d_vec2.CopyFromHost(vec2.data(), n);

        auto d_result = CudaMemRAII<DemoType>::Alloc(n);
        
        // thread num
        sz num_threads = 1024;
        dim3 gridDim((n + num_threads - 1) / num_threads, 1, 1);
        dim3 blockDim(num_threads, 1, 1);

        add <<<gridDim, blockDim>>> (d_vec1.get(), d_vec2.get(), d_result.get(), n);
        cudaDeviceSynchronize();

        d_result.CopyToHost(result.data(), n);
    }

    freopen("output.txt", "w", stdout);
    for (sz i = 0; i < n; i++)
    {
        std::cout << vec1[i] << " + " << vec2[i] << " = " << result[i] << std::endl;
    }
    fclose(stdout);
}

void add2d_demo()
{
    using DemoType = int;
    static constexpr sz Row = 1000;
    static constexpr sz Col = 2;

    auto vec1 = std::unique_ptr<DemoType[]>(new DemoType[Row * Col]);
    auto vec2 = std::unique_ptr<DemoType[]>(new DemoType[Row * Col]);
    auto result = std::unique_ptr<DemoType[]>(new DemoType[Row * Col]);

    for (sz i = 0; i < Row; i++)
    {
        for (sz j = 0; j < Col; j++)
        {
            vec1[i * Col + j] = CommonUtils::random<DemoType>(0, 100);
            vec2[i * Col + j] = CommonUtils::random<DemoType>(0, 100);
        }
    }

    // cuda add

    {
        auto d_vec1 = CudaMemRAII<DemoType>::Alloc(Row * Col);
        auto d_vec2 = CudaMemRAII<DemoType>::Alloc(Row * Col);

        d_vec1.CopyFromHost(vec1.get(), Row * Col);
        d_vec2.CopyFromHost(vec2.get(), Row * Col);

        auto d_result = CudaMemRAII<DemoType>::Alloc(Row * Col);
        
        // thread num
        sz block_size = 8;
        
        dim3 gridDim(block_size, block_size, 1);
        dim3 blockDim(
            (Col + block_size - 1) / block_size,
            (Row + block_size - 1) / block_size, 1);

        add <<<gridDim, blockDim>>> (d_vec1.get(), d_vec2.get(), d_result.get(), Row, Col);
        cudaDeviceSynchronize();

        d_result.CopyToHost(result.get(), Row * Col);
    }

    freopen("output.txt", "w", stdout);
    for (sz i = 0; i < Row; i++)
    {
        for (sz j = 0; j < Col; j++)
        {
            std::cout << std::setw(3) << vec1[i * Col + j] << " + " 
                      << std::setw(3) << vec2[i * Col + j] << " = "
                      << std::setw(3) << result[i * Col + j]
                      << std::setw(6) << " ";
        }
        std::cout << std::endl;
    }
    fclose(stdout);
}


int main(void)
{
    // add_demo();
    add2d_demo();
    cudaDeviceReset();
    return 0;
}