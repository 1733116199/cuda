#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void index_calculation_for_2d_grid(int *input)
{
    int gid = (blockIdx.x * gridDim.x + blockIdx.y) * blockDim.x + threadIdx.x;
    printf("blockIdx.x : %d, blockIdx.y: %d, Idx : %d, value : %d\n", blockIdx.x, blockIdx.y, gid, input[gid]);
}

int main()
{
    int h_data[] = {
        23,
        9,
        4,
        53,
        65,
        12,
        1,
        33,
        87,
        45,
        23,
        12,
        342,
        56,
        44,
        99,
    };

    for (int i = 0; i < sizeof(h_data) / sizeof(int); i++)
    {
        printf("%d ", h_data[i]);
    }
    printf("\n \n");

    int *d_data;
    cudaMalloc((void **)&d_data, sizeof(h_data));
    cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice);

    dim3 block(4);
    dim3 grid(2, 2);
    index_calculation_for_2d_grid<<<grid, block>>>(d_data);

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}