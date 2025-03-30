#ifndef SOFTMAX_H   
#define SOFTMAX_H

void softmax_cpu(const float* input, float* output, int N);

void softmax_gpu(const float* input, float* output, int N);

#ifdef __CUDACC__       
// Complete declarations
__global__ void reduce_max_kernel(const float* input, float* output, int N);
__global__ void reduce_sum_kernel(const float* input, float* output, int N);
__global__ void compute_exp_kernel(const float* input, float* output, float max_val, int N);
__global__ void softmax_kernel(const float* input, float* output, float max_val, float sum, int N);
#endif

#endif    