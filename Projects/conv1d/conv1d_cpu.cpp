#include "conv1d.h" 

// CPU implementation of 1D convolution (no padding)
void conv1d_cpu(const float* input, float* output, const float* kernel, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    
    // Process each output element
    for (int i = 0; i < output_size; i++) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            sum += input[i + k] * kernel[k];
        }
        output[i] = sum;
    }
}

// CPU implementation with zero padding to maintain input size
void conv1d_cpu_padded(const float* input, float* output, const float* kernel, int input_size, int kernel_size) {
    // With proper padding, output size equals input size
    int output_size = input_size;
    int padding = kernel_size / 2;
    
    // Process each output element
    for (int i = 0; i < output_size; i++) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            int idx = i - padding + k;
            if (idx >= 0 && idx < input_size) {
                sum += input[idx] * kernel[k];
            }
            // Zero padding for out-of-bounds access
        }
        output[i] = sum;
    }
}