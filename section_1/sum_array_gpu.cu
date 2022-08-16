#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>
#include <cassert>

__global__ void sum_array_gpu(int *a, int *b, int *c, int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size)
    {
        c[gid] = a[gid] + b[gid];
    }
}

void sum_array_cpu(int *a, int *b, int *c, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

void compare_arrays(int *h_c, int *gpu_results, int size)
{
    for (int i = 0; i < size; i++)
    {
        assert(h_c[i] == gpu_results[i]);
    }
}

int main()
{
    int size = 10000;
    int block_size = 1024;
    int NO_BYTES = size * sizeof(int);

    int *h_a = (int *)malloc(NO_BYTES);
    int *h_b = (int *)malloc(NO_BYTES);
    int *h_c = (int *)malloc(NO_BYTES);
    int *gpu_results = (int *)malloc(NO_BYTES);

    cudaError error;
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        h_a[i] = (int)(rand() & 0xFF);
    }
    for (int i = 0; i < size; i++)
    {
        h_b[i] = (int)(rand() & 0xFF);
    }

    clock_t cpu_start = clock();
    sum_array_cpu(h_a, h_b, h_c, size);
    clock_t cpu_end = clock();

    memset(gpu_results, 0, NO_BYTES);

    int *d_a;
    error = cudaMalloc((int **)&d_a, NO_BYTES);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
        exit(1);
    }
    int *d_b;
    error = cudaMalloc((int **)&d_b, NO_BYTES);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
        exit(1);
    }
    int *d_c;
    cudaMalloc((int **)&d_c, NO_BYTES);

    clock_t h_to_d_start = clock();
    cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);
    clock_t h_to_d_end = clock();

    dim3 block(block_size);
    dim3 grid((size / block.x) + 1);

    clock_t gpu_start = clock();
    sum_array_gpu<<<grid, block>>>(d_a, d_b, d_c, size);
    cudaDeviceSynchronize();
    clock_t gpu_end = clock();

    clock_t d_to_h_start = clock();
    cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);
    clock_t d_to_h_end = clock();

    compare_arrays(h_c, gpu_results, size);

    printf("Sum array CPU execution time: %4.6f \n", ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC);
    printf("Host to Device mem transfer time: %4.6f \n", ((double)(h_to_d_end - h_to_d_start)) / CLOCKS_PER_SEC);
    printf("Device to Host mem transfer time: %4.6f \n", ((double)(d_to_h_end - d_to_h_start)) / CLOCKS_PER_SEC);
    printf("Sum array GPU kernel execution time: %4.6f \n", ((double)(gpu_end - gpu_start)) / CLOCKS_PER_SEC);
    printf("Sum array Total GPU execution time: %4.6f \n", ((double)(h_to_d_end - h_to_d_start + d_to_h_end - d_to_h_start + gpu_end - gpu_start)) / CLOCKS_PER_SEC);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);
    free(gpu_results);

    cudaDeviceReset();
    return 0;
}