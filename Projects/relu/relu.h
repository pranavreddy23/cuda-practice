#ifndef RELU_H
#define RELU_H

void relu_cpu(const float* input, float* output, int n, int m);
void relu_cpu_avx(const float* input, float* output, int n, int m);
extern "C" void relu_gpu(const float* input, float* output, int n, int m);

#ifdef __CUDACC__
__global__ void relu_kernel(const float* input, float* output, int n, int m);
__global__ void relu_kernel_vector4(const float4* input, float4* output, int total_elements);
#endif

#endif // RELU_H