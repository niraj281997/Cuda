#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void f_device()
{
    printf("Hello from the thread\n");
}

int main()
{
    f_device<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}

