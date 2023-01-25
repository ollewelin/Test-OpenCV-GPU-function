// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h> //dynamical 
// Include associated header file.
#include "cuda_kernel.cuh"
#include <stdio.h>



__global__ void dotProductKernel(double* A, double* B, double* result) {
    // number of elements per thread block
    const int N = 256;

    // shared memory for storing partial sums
    __shared__ double sums[N];

    // thread index
    int i = threadIdx.x;

    // partial dot product
    double temp = 0;
    for (int j = i; j < N; j += N) {
        temp += A[j] * B[j];
    }

    // store partial sum in shared memory
    sums[i] = temp;

    // synchronize threads in block
    __syncthreads();

    // reduction in shared memory
    for (int offset = 1; offset < N; offset *= 2) {
        if (i % (2*offset) == 0) {
            sums[i] += sums[i + offset];
        }
        __syncthreads();
    }
   // result[i] = sums[i];
    // write result for block to global memory
    if (i == 0) {
        result[blockIdx.x] = sums[0];
   }
}



void kernel(double *A, double *B, double *Result, int arraySize) {
    // Initialize device pointers.
    double* gpu_A;
    double* gpu_B;
    double* gpu_result;


    // Always Allocate new device memory for A every time I call this fuction
    cudaMalloc((void**) &gpu_A, arraySize * sizeof(double));
    cudaMalloc((void**) &gpu_B, arraySize * sizeof(double));
    cudaMalloc((void**) &gpu_result, sizeof(double));

    // Transfer arrays A and B to device.
    cudaMemcpy(gpu_A, A, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, arraySize * sizeof(double), cudaMemcpyHostToDevice);
  //  cudaMemcpy(gpu_result, Result, arraySize * sizeof(double), cudaMemcpyHostToDevice);
   
    dim3 threads(256);
    dim3 blocks(1);
    dotProductKernel<<<blocks,threads>>>(gpu_A,gpu_B, gpu_result);

    cudaMemcpy(Result, gpu_result,  sizeof(double), cudaMemcpyDeviceToHost);
   // Result[1] = 44;
    // Free device memory 
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_result);
}

