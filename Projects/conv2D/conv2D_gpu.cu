
#include <cuda_runtime.h>
// CUDA kernel for 2D convolution
__global__ void convolutionKernel(const float* input, const float* kernel, float* output,
                                 int input_rows, int input_cols, 
                                 int kernel_rows, int kernel_cols) {
    // Calculate output position
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate output dimensions
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    
    // Check if thread is within output bounds
    if (row < output_rows && col < output_cols) {
        float sum = 0.0f;
        
        // Apply convolution at this position
        for (int k_row = 0; k_row < kernel_rows; ++k_row) {
            for (int k_col = 0; k_col < kernel_cols; ++k_col) {
                int input_row = row + k_row;
                int input_col = col + k_col;
                
                sum += input[input_row * input_cols + input_col] * 
                       kernel[k_row * kernel_cols + k_col];
            }
        }
        
        // Write the result to output
        output[row * output_cols + col] = sum;
    }
}




void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    
    // Calculate output dimensions
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    
    // Define block and grid dimensions
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim(
        (output_cols + blockDim.x - 1) / blockDim.x,
        (output_rows + blockDim.y - 1) / blockDim.y
    );
    
    // Launch the kernel
    convolutionKernel<<<gridDim, blockDim>>>(
        input, kernel, output,
        input_rows, input_cols,
        kernel_rows, kernel_cols
    );
    
    // Wait for the kernel to complete (useful for error checking)
    cudaDeviceSynchronize();
}