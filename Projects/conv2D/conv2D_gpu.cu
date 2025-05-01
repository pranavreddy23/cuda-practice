#include <cuda_runtime.h>
#include "../common/profiler.h"
#include "conv2D.h"
#include <algorithm> // For max/min
#include <iostream>
using namespace std;

// CUDA kernel assuming 'input' is pre-padded - FIXED VERSION
__global__ void convolutionKernelPadded(unsigned char* input, unsigned char* kernel, unsigned char* output,
                                     int original_rows, int original_cols,
                                     int kernel_rows, int kernel_cols, int channels) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if within output bounds
    if (x < original_cols && y < original_rows) {
        // Calculate padding
        int pad_rows = (kernel_rows - 1) / 2;
        int pad_cols = (kernel_cols - 1) / 2;
        int padded_width = original_cols + 2 * pad_cols;
        
        // For each channel
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            float kernel_sum = 0.0f;
            
            // Apply the kernel - CORRECT COORDINATES
            for (int ky = 0; ky < kernel_rows; ky++) {
                for (int kx = 0; kx < kernel_cols; kx++) {
                    // Calculate position in padded input - FIXED
                    // Add pad_rows/pad_cols to account for padding offset
                    int input_y = y + ky;
                    int input_x = x + kx;
                    
                    // Get kernel value (normalized from 0-255)
                    float kernel_val = kernel[ky * kernel_cols + kx] / 255.0f;
                    
                    // Get input value from padded input
                    float input_val = input[(input_y * padded_width + input_x) * channels + c];
                    
                    // Accumulate weighted sum
                    sum += input_val * kernel_val;
                    kernel_sum += kernel_val;
                }
            }
            
            // Normalize by kernel sum if not zero
            if (kernel_sum > 0.0f) {
                sum /= kernel_sum;
            }
            
            // Clamp to 0-255 range
            sum = fmaxf(0.0f, fminf(255.0f, sum));
            
            // Write to output - IMPORTANT: output is NOT padded
            output[(y * original_cols + x) * channels + c] = static_cast<unsigned char>(sum);
        }
    }
}
// Wrapper function to launch the padded kernel
// Takes original dimensions, kernel dimensions. Assumes input/output are pre-allocated correctly.
extern "C" void conv2D_gpu(unsigned char* input, unsigned char* kernel, unsigned char* output,
                         int original_rows, int original_cols, int kernel_rows, int kernel_cols, int channels) {

    Profiler::GPUTimer kernel_timer;
    static Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();

    // Padding size
    int pad_rows = (kernel_rows - 1) / 2;
    int pad_cols = (kernel_cols - 1) / 2;

    // Calculate sizes
    int padded_rows = original_rows + 2 * pad_rows;
    int padded_cols = original_cols + 2 * pad_cols;
    size_t padded_input_size = static_cast<size_t>(padded_rows) * padded_cols * channels * sizeof(unsigned char);
    size_t kernel_size_bytes = static_cast<size_t>(kernel_rows) * kernel_cols * sizeof(unsigned char);
    size_t output_size_bytes = static_cast<size_t>(original_rows) * original_cols * channels * sizeof(unsigned char);


    // Allocate GPU memory
    unsigned char *d_padded_input, *d_kernel, *d_output;
    cudaMalloc(&d_padded_input, padded_input_size);
    cudaMalloc(&d_kernel, kernel_size_bytes);
    cudaMalloc(&d_output, output_size_bytes);
    mem_tracker.record_gpu_allocation(padded_input_size + kernel_size_bytes + output_size_bytes);


    // Copy data from host to device
    cudaMemcpy(d_padded_input, input, padded_input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size_bytes, cudaMemcpyHostToDevice);


    // Define block and grid dimensions based on *original* output size
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (original_cols + blockDim.x - 1) / blockDim.x,
        (original_rows + blockDim.y - 1) / blockDim.y
    );
    
    // Launch the kernel
    kernel_timer.start();
    convolutionKernelPadded<<<gridDim, blockDim>>>(
        d_padded_input, d_kernel, d_output,
        original_rows, original_cols,
        kernel_rows, kernel_cols, channels
    );
    cudaDeviceSynchronize(); // Wait for kernel completion
    kernel_timer.stop();
    Profiler::KernelTimeTracker::record_kernel_time("conv2D_padded", kernel_timer.elapsed_milliseconds()); // Use a distinct name


    // Copy result back from device to host
    cudaMemcpy(output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_padded_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    // Optional: Add cudaGetLastError checks after CUDA calls for debugging
}