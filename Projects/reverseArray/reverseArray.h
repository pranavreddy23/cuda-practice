#ifndef REVERSE_ARRAY_H
#define REVERSE_ARRAY_H

void reverseArray_cpu(float* input, int N);
void reverseArray_gpu(float* input, int N);

#ifdef __CUDACC__
__global__ void reverse_array(float* input, int N);
#endif

#endif