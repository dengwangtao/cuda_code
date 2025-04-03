#include <stdio.h>
#include <vector>
#include "common.cuh"
#include "add.cuh"
#include "cuda_utils.cuh"


void add_demo()
{
    using DemoType = int;
    sz n = 10000;

    std::vector<DemoType> vec1;
    std::vector<DemoType> vec2;
    std::vector<DemoType> result(n);

    for (sz i = 0; i < n; i++)
    {
        vec1.push_back(CommonUtils::random(0, 100));
        vec2.push_back(CommonUtils::random(0, 100));
    }

    // cuda add

    {
        auto d_vec1 = CudaMemRAII<DemoType>::Alloc(n);
        auto d_vec2 = CudaMemRAII<DemoType>::Alloc(n);

        d_vec1.CopyFromHost(vec1.data(), n);
        d_vec2.CopyFromHost(vec2.data(), n);

        auto d_result = CudaMemRAII<DemoType>::Alloc(n);

        add <<<(n + 1023) / 1024, 1024>>> (d_vec1.get(), d_vec2.get(), d_result.get(), n);
        cudaDeviceSynchronize();

        d_result.CopyToHost(result.data(), n);
    }

    freopen("output.txt", "w", stdout);
    for (sz i = 0; i < n; i++)
    {
        printf("%d + %d = %d\n", vec1[i], vec2[i], result[i]);
    }

    cudaDeviceReset();
}

int main(void)
{
    add_demo();
    return 0;
}