// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h> //dynamical 
// Include associated header file.
#include "cuda_kernel.cuh"
#include <stdio.h>



__global__ void dotProductKernel(double* A, double* B, double* result) {
    // number of elements per thread block
    int N = blockDim.x;

    // shared memory for storing partial sums
    __shared__ double sums[128];

    // thread index
    int ix = threadIdx.x;
    int bx = blockIdx.x;

    // partial dot product
    double temp = 0;
    for (int j = ix; j < N; j += N) {
        temp += A[j + N*bx] * B[j + N*bx];
    }

    // store partial sum in shared memory
    sums[threadIdx.x] = temp;

    // synchronize threads in block
    __syncthreads();

    // reduction in shared memory
    for (int offset = 1; offset < N; offset *= 2) {
        if (threadIdx.x % (2*offset) == 0) {
            sums[threadIdx.x] += sums[threadIdx.x + offset];
        }
        __syncthreads();
    }
   // result[i] = sums[i];
    // write result for block to global memory
    if (threadIdx.x == 0) {
        result[blockIdx.x] = sums[0];
   }
}



void kernel(double *A, double *B, double *Result, int arraySize) {
    // Initialize device pointers.
    const int block_n = 2;
    double* gpu_A;
    double* gpu_B;
    double* gpu_result;


    // Always Allocate new device memory for A every time I call this fuction
    cudaMalloc((void**) &gpu_A, arraySize * sizeof(double));
    cudaMalloc((void**) &gpu_B, arraySize * sizeof(double));
    cudaMalloc((void**) &gpu_result, block_n * sizeof(double));

    // Transfer arrays A and B to device.
    cudaMemcpy(gpu_A, A, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, arraySize * sizeof(double), cudaMemcpyHostToDevice);
  //  cudaMemcpy(gpu_result, Result, arraySize * sizeof(double), cudaMemcpyHostToDevice);
   
    dim3 threads(128);
    dim3 blocks(block_n);
    dotProductKernel<<<blocks,threads>>>(gpu_A,gpu_B, gpu_result);

    cudaMemcpy(Result, gpu_result, block_n * sizeof(double), cudaMemcpyDeviceToHost);
   // Result[1] = 44;
    // Free device memory 
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_result);
}

