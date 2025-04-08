#include <iostream>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(float *d_input, float *d_output, float *d_kernel, int input_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Skip out-of-bound indices
    if (idx >= input_size - kernel_size + 1) return;

    float result = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        result += d_input[idx + i] * d_kernel[i];
    }
    d_output[idx] = result;
}

void conv1d(float *input, float *output, float *kernel, int input_size, int kernel_size) {
    static Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    Profiler::GPUTimer total_timer;
    Profiler::GPUTimer kernel_timer;
    total_timer.start();
    float *d_input, *d_output, *d_kernel;
    size_t input_size_bytes = input_size * sizeof(float);
    size_t kernel_size_bytes = kernel_size * sizeof(float);
    size_t output_size_bytes = (input_size - kernel_size + 1) * sizeof(float);

    cudaMalloc((void**)&d_input, input_size_bytes);
    cudaMalloc((void**)&d_output, output_size_bytes);
    cudaMalloc((void**)&d_kernel, kernel_size_bytes);
    mem_tracker.record_gpu_allocation(input_size_bytes + output_size_bytes + kernel_size_bytes);
    cudaMemcpy(d_input, input, input_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size_bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (input_size - kernel_size + 1 + threads_per_block - 1) / threads_per_block;

    kernel_timer.start();
    conv1d_kernel<<<blocks, threads_per_block>>>(d_input, d_output, d_kernel, input_size, kernel_size);
    kernel_timer.stop();
    Profiler::KernelTimeTracker::record_kernel_time("conv1d", kernel_timer.elapsed_milliseconds());

    cudaMemcpy(output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    total_timer.stop();
    Profiler::KernelTimeTracker::last_total_time = total_timer.elapsed_milliseconds();
}

