#ifndef CONV1D_H
#define CONV1D_H

void conv1d_cpu(const float* input, const float* kernel, float* output, int input_size, int kernel_size);
void conv1d_gpu(const float* input, const float* kernel, float* output, int input_size, int kernel_size);
// CUDA kernel declaration (implemented in conv1d_gpu.cu)
#ifdef __CUDACC__
__global__ void conv1d_kernel(float *d_input, float *d_output, float *d_kernel, int input_size, int kernel_size, int output_size);
#endif

#endif