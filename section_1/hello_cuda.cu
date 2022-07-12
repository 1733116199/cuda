#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_cuda()
{
    printf("Hello CUDA world! \n");
}

int main()
{
    dim3 block(8, 2, 1);
    dim3 grid(2, 2, 1);
    hello_cuda<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}