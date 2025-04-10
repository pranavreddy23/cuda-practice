#ifndef CONV1D_H
#define CONV1D_H

// GPU implementation of 1D convolution
void conv1d_gpu(float *input, float *output, float *kernel, int input_size, int kernel_size);

// CPU implementation of 1D convolution
void conv1d_cpu(const float* input, float* output, const float* kernel, int input_size, int kernel_size);

// CPU implementation with zero padding for direct comparison with GPU
void conv1d_cpu_padded(const float* input, float* output, const float* kernel, int input_size, int kernel_size);

#endif // CONV1D_H