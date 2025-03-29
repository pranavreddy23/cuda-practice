#ifndef MAT_TRANSPOSE_H
#define MAT_TRANSPOSE_H

void matrix_transpose_cpu(const float* input, float* output, int rows, int cols);
void matrix_transpose_gpu(const float* input, float* output, int rows, int cols);

#ifdef __CUDACC__
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols);
#endif

#endif