#ifndef REVERSE_ARRAY_H
#define REVERSE_ARRAY_H

// CPU implementation
void reverseArray_cpu(float* input, int N);

// GPU implementation
void reverseArray_gpu(float* input, int N);

// AVX2 SIMD implementation
void reverseArray_avx(float* input, int N);

#ifdef __CUDACC__
__global__ void reverse_array(float* input, int N);
#endif

#endif