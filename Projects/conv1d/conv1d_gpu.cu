#include <iostream>
#include <cuda_runtime.h>
#include "../common/profiler.h"
#include "conv1d.h"

__global__ void conv1d_kernel(float *d_input, float *d_output, float *d_kernel, int input_size, int kernel_size, int output_size) {
    extern __shared__ float shared_input[];
    
    int block_size = blockDim.x;
    int halo_radius = kernel_size / 2;
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate input indices for this thread
    int input_start_idx = blockIdx.x * blockDim.x - halo_radius;
    
    // Load regular elements
    if (input_start_idx + tid >= 0 && input_start_idx + tid < input_size) {
        shared_input[tid] = d_input[input_start_idx + tid];
    } else {
        shared_input[tid] = 0.0f; // Zero padding for out-of-bounds
    }
    
    // Load halo elements
    if (tid < 2 * halo_radius) {
        int halo_idx = input_start_idx + tid + block_size;
        if (halo_idx >= 0 && halo_idx < input_size) {
            shared_input[tid + block_size] = d_input[halo_idx];
        } else {
            shared_input[tid + block_size] = 0.0f; // Zero padding for out-of-bounds
        }
    }
    
    __syncthreads();
    
    // Compute convolution only if this thread corresponds to a valid output element
    if (gid < output_size) {
        float result = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            result += shared_input[tid + halo_radius + k] * d_kernel[k];
        }
        d_output[gid] = result;
    }
}

void conv1d_gpu(float *input, float *output, float *kernel, int input_size, int kernel_size) {
    static Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    Profiler::GPUTimer total_timer;
    Profiler::GPUTimer kernel_timer;
    
    total_timer.start();
    
    // Calculate output size
    int output_size = input_size - kernel_size + 1;
    
    // Allocate device memory
    float *d_input, *d_output, *d_kernel;
    size_t input_size_bytes = input_size * sizeof(float);
    size_t kernel_size_bytes = kernel_size * sizeof(float);
    size_t output_size_bytes = output_size * sizeof(float);
    
    cudaMalloc((void**)&d_input, input_size_bytes);
    cudaMalloc((void**)&d_output, output_size_bytes);
    cudaMalloc((void**)&d_kernel, kernel_size_bytes);
    mem_tracker.record_gpu_allocation(input_size_bytes + output_size_bytes + kernel_size_bytes);
    
    // Copy data to device
    cudaMemcpy(d_input, input, input_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size_bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (output_size + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = (threads_per_block + kernel_size - 1) * sizeof(float);
    
    kernel_timer.start();
    conv1d_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_input, d_output, d_kernel, input_size, kernel_size, output_size);
    cudaDeviceSynchronize();
    kernel_timer.stop();
    Profiler::KernelTimeTracker::record_kernel_time("conv1d", kernel_timer.elapsed_milliseconds());
    
    // Copy result back to host
    cudaMemcpy(output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    
    total_timer.stop();
    Profiler::KernelTimeTracker::last_total_time = total_timer.elapsed_milliseconds();
}

