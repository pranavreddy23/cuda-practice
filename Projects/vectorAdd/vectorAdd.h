#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#include <vector>

// CPU implementation of vector addition
void vectorAdd_cpu(const std::vector<int>& a, const std::vector<int>& b,
                  std::vector<int>& c, int N);

// GPU implementation of vector addition
void vectorAdd_gpu(const std::vector<int>& a, const std::vector<int>& b,
                  std::vector<int>& c, int N);

// CUDA kernel declaration (implemented in vectorAdd_gpu.cu)
#ifdef __CUDACC__
__global__ void vectorAdd_kernel(const int *__restrict a, const int *__restrict b,
                                int *__restrict c, int N);
#endif

#endif // VECTOR_ADD_H
