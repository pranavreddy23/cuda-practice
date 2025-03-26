#include "matTranspose.h"
#include <cuda_runtime.h>
#include "../common/profiler.h"

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// input, output are host pointers (i.e. pointers to memory on the CPU)
void matrix_transpose_gpu(const float* input, float* output, int rows, int cols) {
    static Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    Profiler::GPUTimer total_timer;
    Profiler::GPUTimer kernel_timer;
    
    total_timer.start();
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * rows * cols);
    cudaMalloc(&d_output, sizeof(float) * rows * cols);
    
    // Track GPU memory allocation
    mem_tracker.record_gpu_allocation(sizeof(float) * rows * cols * 2);
    
    // Copy input data from host to device
    cudaMemcpy(d_input, input, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockDim(16,16); // Define a block size (16x16 threads per block)
    dim3 gridDim(
        (rows + blockDim.x - 1) / blockDim.x, 
        (cols + blockDim.y - 1) / blockDim.y
    ); // Grid size to cover entire matrix

    // Start kernel timer
    kernel_timer.start();
    
    // Launch kernel
    matrix_transpose_kernel<<<gridDim, blockDim>>>(d_input, d_output, rows, cols);
    
    // Stop kernel timer
    kernel_timer.stop();
    Profiler::KernelTimeTracker::last_kernel_time = kernel_timer.elapsed_milliseconds();

    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Copy result from device to host
    cudaMemcpy(output, d_output, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Stop total timer
    total_timer.stop();
    Profiler::KernelTimeTracker::last_total_time = total_timer.elapsed_milliseconds();
}