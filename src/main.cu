#include <stdio.h>

__global__
void say_hello(void)
{
    printf("Hello, world!\n");
}


int main(void)
{

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    say_hello <<<1,10>>>();
    cudaDeviceReset();

    printf("Hello, world! Done.!\n");
    return 0;
}