#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void unique_idx_calc_threadIdx(int *input)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Idx : %d, value : %d\n", gid, input[gid]);
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
    dim3 grid(4);
    unique_idx_calc_threadIdx<<<grid, block>>>(d_data);

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}