#include "matmul.h"
#include <immintrin.h>

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    // Outer loops remain the same, but inner computation uses SIMD
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            // Use AVX 256-bit registers (8 floats at a time)
            __m256 sum = _mm256_setzero_ps();
            
            // Process 8 elements at a time
            int k;
            for (k = 0; k + 7 < N; k += 8) {
                // Load 8 elements from A and B
                __m256 a_vec = _mm256_loadu_ps(&A[i * N + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[k * K + j]);
                
                // Multiply and accumulate
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
            }
            
            // Horizontal sum of the vector
            __m128 sum_128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), 
                                         _mm256_extractf128_ps(sum, 1));
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            float total_sum = _mm_cvtss_f32(sum_128);
            
            // Handle remaining elements
            for (; k < N; k++) {
                total_sum += A[i * N + k] * B[k * K + j];
            }
            
            // Store the result
            C[i * K + j] = total_sum;
        }
    }
}