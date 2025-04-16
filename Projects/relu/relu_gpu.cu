#include "relu.h"
#include <cuda_runtime.h>
#include "../common/profiler.h"

// Optimized ReLU kernel using vectorized loads/stores for greater memory throughput
__global__ void relu_kernel(const float* input, float* output, int n, int m) {
    // Calculate thread position (row-major indexing)
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within bounds
    if (idx < n * m) {
        // Apply ReLU: max(0, x)
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

// Vector type for processing 4 elements at once
__global__ void relu_kernel_vector4(const float4* input, float4* output, int total_elements) {
    // Calculate thread position for vector4 processing
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes 4 elements at once
    if (idx < total_elements) {
        // Load vector of 4 values
        float4 in4 = input[idx];
        
        // Apply ReLU to each component
        float4 out4;
        out4.x = (in4.x > 0.0f) ? in4.x : 0.0f;
        out4.y = (in4.y > 0.0f) ? in4.y : 0.0f;
        out4.z = (in4.z > 0.0f) ? in4.z : 0.0f;
        out4.w = (in4.w > 0.0f) ? in4.w : 0.0f;
        
        // Store result
        output[idx] = out4;
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void relu_gpu(const float* input, float* output, int n, int m) {
    static Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    Profiler::GPUTimer kernel_timer;
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * n * m);
    cudaMalloc(&d_output, sizeof(float) * n * m);

    mem_tracker.record_gpu_allocation(sizeof(float) * n * m);

    cudaMemcpy(d_input, input, sizeof(float) * n * m, cudaMemcpyHostToDevice);

    const int total = n * m;
    
    // Check if we can use vectorized approach (matrix size divisible by 4)
    if (total % 4 == 0) {
        // Reinterpret pointers as float4
        const float4* input4 = reinterpret_cast<const float4*>(d_input);
        float4* output4 = reinterpret_cast<float4*>(d_output);
        
        // Calculate grid dimensions
        const int total_vec4 = total / 4;
        const int blockSize = 256;
        const int gridSize = (total_vec4 + blockSize - 1) / blockSize;

        // Only time the kernel execution
        kernel_timer.start();
        // Launch vectorized kernel
        relu_kernel_vector4<<<gridSize, blockSize>>>(input4, output4, total_vec4);
        cudaDeviceSynchronize();
        kernel_timer.stop();
        Profiler::KernelTimeTracker::record_kernel_time("relu_vector4", kernel_timer.elapsed_milliseconds());
    } else {
        // Use standard approach for non-multiple-of-4 sizes
        const int blockSize = 256;
        const int gridSize = (total + blockSize - 1) / blockSize;

        // Only time the kernel execution
        kernel_timer.start();
        // Launch standard kernel
        relu_kernel<<<gridSize, blockSize>>>(d_input, d_output, n, m);
        cudaDeviceSynchronize();
        kernel_timer.stop();
        Profiler::KernelTimeTracker::record_kernel_time("relu", kernel_timer.elapsed_milliseconds());
    }

    cudaMemcpy(output, d_output, sizeof(float) * n * m, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaDeviceSynchronize();

    
}