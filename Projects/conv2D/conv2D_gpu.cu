#include <cuda_runtime.h>
#include "../common/profiler.h"
#include "conv2D.h"
#include <algorithm> // For max/min

// CUDA kernel assuming 'input' is pre-padded
__global__ void convolutionKernelPadded(unsigned char* input, unsigned char* kernel, unsigned char* output,
                                     int original_rows, int original_cols,
                                     int kernel_rows, int kernel_cols) {

    // Calculate thread's corresponding position in the *original* image (output grid)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Padding size
    int pad_rows = (kernel_rows - 1) / 2;
    int pad_cols = (kernel_cols - 1) / 2;

    // Dimensions of the padded input
    int padded_cols = original_cols + 2 * pad_cols;

    // Check if thread is within the bounds of the *original* image
    if (row < original_rows && col < original_cols) {
        int sum = 0;

        // Apply convolution kernel
        for (int k_row = 0; k_row < kernel_rows; ++k_row) {
            for (int k_col = 0; k_col < kernel_cols; ++k_col) {
                // Calculate corresponding coordinates in the *padded* input
                int input_row = row + k_row; // Index relative to top-left of padded input
                int input_col = col + k_col;

                // We assume the caller padded correctly, so bounds check might be optional
                // but good for safety if kernel could go out of bounds theoretically
                // (though gridDim calculation should prevent this if done right)
                 sum += input[input_row * padded_cols + input_col] *
                        kernel[k_row * kernel_cols + k_col];

            }
        }

        // Write the result to the output buffer (sized as original image)
        // Clamp result
        output[row * original_cols + col] = static_cast<unsigned char>(max(0, min(255, sum)));
    }
}

// Wrapper function to launch the padded kernel
// Takes original dimensions, kernel dimensions. Assumes input/output are pre-allocated correctly.
extern "C" void conv2D_gpu(unsigned char* input, unsigned char* kernel, unsigned char* output,
                         int original_rows, int original_cols, int kernel_rows, int kernel_cols) {

    Profiler::GPUTimer kernel_timer;
    static Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();

    // Padding size
    int pad_rows = (kernel_rows - 1) / 2;
    int pad_cols = (kernel_cols - 1) / 2;

    // Calculate sizes
    int padded_rows = original_rows + 2 * pad_rows;
    int padded_cols = original_cols + 2 * pad_cols;
    size_t padded_input_size = static_cast<size_t>(padded_rows) * padded_cols * sizeof(unsigned char); // Assuming 1 channel for now
    size_t kernel_size_bytes = static_cast<size_t>(kernel_rows) * kernel_cols * sizeof(unsigned char);
    size_t output_size_bytes = static_cast<size_t>(original_rows) * original_cols * sizeof(unsigned char);


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
        kernel_rows, kernel_cols
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