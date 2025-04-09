#include "matmul.h"
#include <cuda_runtime.h>
#include "../common/profiler.h"
#define TILE_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Calculate global thread coordinates
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for(int j = 0; j < (N + TILE_SIZE - 1) / TILE_SIZE; ++j) {
        if(row < M && col < K && j * TILE_SIZE + tx < N) {
            tile_a[ty][tx] = A[row * N + j * TILE_SIZE + tx];
        }else{
            tile_a[ty][tx] = 0.0f;
        }
        if(row < M && col < K && j * TILE_SIZE + ty < N) {
            tile_b[ty][tx] = B[(j * TILE_SIZE + ty) * K + col];
        }else{
            tile_b[ty][tx] = 0.0f;
        }
        __syncthreads();
        for(int z = 0; z < TILE_SIZE; ++z) {
            if(col < K && j * TILE_SIZE + z < N) {
                C[row * K + col] += tile_a[ty][z] * tile_b[z][tx];
            }
        }
        __syncthreads();
    }
    
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K) {
    
    static Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    Profiler::GPUTimer total_timer;
    Profiler::GPUTimer kernel_timer;
    total_timer.start();
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sizeof(float) * M * N);
    cudaMalloc(&d_b, sizeof(float) * N * K);
    cudaMalloc(&d_c, sizeof(float) * M * K);
    mem_tracker.record_gpu_allocation(sizeof(float) * M * N + sizeof(float) * N * K + sizeof(float) * M * K);
    
    cudaMemcpy(d_a, A, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(float) * N * K, cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim(
        (K + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
    );
    
    kernel_timer.start();
    matmul_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    kernel_timer.stop();
    Profiler::KernelTimeTracker::record_kernel_time("matmul", kernel_timer.elapsed_milliseconds());

    cudaMemcpy(C, d_c, sizeof(float) * M * K, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    total_timer.stop();
    Profiler::KernelTimeTracker::last_total_time = total_timer.elapsed_milliseconds();
}