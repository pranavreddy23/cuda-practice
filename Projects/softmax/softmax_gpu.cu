
#include <cuda_runtime.h>
#include <float.h>
#include "../common/profiler.h"
#include "softmax.h"

// Kernel for parallel reduction of max
__global__ void reduce_max_kernel(const float* input, float* output, int N) {
    __shared__ float shared_data[256];
    
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load data into shared memory
    shared_data[tid] = (gid < N) ? input[gid] : -FLT_MAX;
    __syncthreads();
    
    // Reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    
    // First thread writes block's max
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

// Kernel for parallel reduction of sum
__global__ void reduce_sum_kernel(const float* input, float* output, int N) {
    __shared__ float shared_data[256];
    
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load data into shared memory
    shared_data[tid] = (gid < N) ? input[gid] : 0.0f;
    __syncthreads();
    
    // Reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread writes block's sum
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

// Kernel for softmax computation
__global__ void softmax_kernel(const float* input, float* output, float max_val, float sum, int N) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (gid < N) {
        float exp_val = expf(input[gid] - max_val);
        output[gid] = exp_val / sum;
    }
}
// Separate exponential computation kernel (to be defined externally)
__global__ void compute_exp_kernel(const float* input, float* output, float max_val, int N) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (gid < N) {
        output[gid] = expf(input[gid] - max_val);
    }
}

void softmax_gpu(const float* input, float* output, int N) {
    static Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    
    // Start total timer
    Profiler::GPUTimer total_timer;
    total_timer.start();
    
    // Block and grid configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Device memory pointers
    float *d_input = NULL, *d_output = NULL, *d_max = NULL, *d_sum = NULL;
    float max_val = 0.0f, total_sum = 0.0f;
    
    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_max, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_sum, blocksPerGrid * sizeof(float));
    mem_tracker.record_gpu_allocation(sizeof(float) * N + sizeof(float) * N + sizeof(float) * blocksPerGrid + sizeof(float) * blocksPerGrid);
    // Copy input to device
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Your kernel executions with individual timings
    Profiler::GPUTimer kernel_timer;
    
    // Find maximum value
    kernel_timer.start();
    reduce_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_max, N);
    reduce_max_kernel<<<1, blocksPerGrid>>>(d_max, d_max, blocksPerGrid);
    kernel_timer.stop();
    Profiler::KernelTimeTracker::record_kernel_time("Reduce Max", kernel_timer.elapsed_milliseconds());
    cudaMemcpy(&max_val, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute exponentials
    kernel_timer.start();
    compute_exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, max_val, N);
    kernel_timer.stop();
    Profiler::KernelTimeTracker::record_kernel_time("Compute Exp", kernel_timer.elapsed_milliseconds());
    
    // Compute sum of exponentials
    kernel_timer.start();
    reduce_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_sum, N);
    reduce_sum_kernel<<<1, blocksPerGrid>>>(d_sum, d_sum, blocksPerGrid);
    kernel_timer.stop();
    Profiler::KernelTimeTracker::record_kernel_time("Reduce Sum", kernel_timer.elapsed_milliseconds());
    cudaMemcpy(&total_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Final softmax normalization
    kernel_timer.start();
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, max_val, total_sum, N);
    kernel_timer.stop();
    Profiler::KernelTimeTracker::record_kernel_time("Softmax Norm", kernel_timer.elapsed_milliseconds());
    
    // Copy result back to host
    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_max);
    cudaFree(d_sum);
    
    // Stop total timer and record
    total_timer.stop();
    Profiler::KernelTimeTracker::last_total_time = total_timer.elapsed_milliseconds();
}
