#include "relu.h"
#include <immintrin.h>

void relu_cpu(const float* input, float* output, int n, int m) {
    for (int i = 0; i < n * m; i++) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

void relu_cpu_avx(const float* input, float* output, int n, int m) {
    for (int i = 0; i < n * m; i += 8) {
        __m256 in4 = _mm256_loadu_ps(&input[i]);
        __m256 out4 = _mm256_max_ps(in4, _mm256_setzero_ps());
        _mm256_storeu_ps(&output[i], out4);
    }
}       