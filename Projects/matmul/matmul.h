#ifndef MATMUL_H
#define MATMUL_H

// // CPU implementation of matrix multiplication
// void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K);

// GPU implementation of matrix multiplication
void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K);

// CPU implementation of matrix multiplication using SIMD
void matmul_cpu_simd(const float* A, const float* B, float* C, int M, int N, int K);

// CPU implementation of matrix multiplication using naive approach
void matmul_cpu_naive(const float* A, const float* B, float* C, int M, int N, int K);

// CUDA kernel declaration (implemented in matlmul_gpu.cu)
#ifdef __CUDACC__
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K);
#endif

#endif // MATLMUL_H
   