#include "matmul.h"
#include <cuda_runtime.h>
#include "../common/profiler.h"

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Calculate global thread coordinates
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within matrix dimensions
    if (row < M && col < K) {
        float sum = 0.0f;
        
        // Perform dot product for this element
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        
        // Store result in output matrix
        C[row * K + col] = sum;
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