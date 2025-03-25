#include "matmul.h"

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            C[i * K + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * K + j] += A[i * N + k] * B[k * K + j];
            }
        }
    }
}