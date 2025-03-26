#include "matTranspose.h"
#include <immintrin.h>

void matrix_transpose_cpu(const float* input, float* output, int rows, int cols) {
    // For small matrices or when dimensions aren't multiples of 8,
    // we'll use a simple approach
    if (rows < 8 || cols < 8 || (rows % 8) != 0 || (cols % 8) != 0) {
        // Simple transpose without SIMD
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[j * rows + i] = input[i * cols + j];
            }
        }
        return;
    }

    // For larger matrices that are multiples of 8, use SIMD
    // Process 8x8 blocks at a time
    for (int i = 0; i < rows; i += 8) {
        for (int j = 0; j < cols; j += 8) {
            // Load 8 rows, each with 8 elements
            __m256 row0 = _mm256_loadu_ps(&input[(i + 0) * cols + j]);
            __m256 row1 = _mm256_loadu_ps(&input[(i + 1) * cols + j]);
            __m256 row2 = _mm256_loadu_ps(&input[(i + 2) * cols + j]);
            __m256 row3 = _mm256_loadu_ps(&input[(i + 3) * cols + j]);
            __m256 row4 = _mm256_loadu_ps(&input[(i + 4) * cols + j]);
            __m256 row5 = _mm256_loadu_ps(&input[(i + 5) * cols + j]);
            __m256 row6 = _mm256_loadu_ps(&input[(i + 6) * cols + j]);
            __m256 row7 = _mm256_loadu_ps(&input[(i + 7) * cols + j]);

            // Transpose 8x8 block using AVX2 intrinsics
            // First, perform 4x4 transposes within the 8x8 block
            __m256 t0 = _mm256_unpacklo_ps(row0, row1);
            __m256 t1 = _mm256_unpackhi_ps(row0, row1);
            __m256 t2 = _mm256_unpacklo_ps(row2, row3);
            __m256 t3 = _mm256_unpackhi_ps(row2, row3);
            __m256 t4 = _mm256_unpacklo_ps(row4, row5);
            __m256 t5 = _mm256_unpackhi_ps(row4, row5);
            __m256 t6 = _mm256_unpacklo_ps(row6, row7);
            __m256 t7 = _mm256_unpackhi_ps(row6, row7);

            __m256 tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
            __m256 tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
            __m256 tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
            __m256 tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

            // Store the transposed data
            _mm256_storeu_ps(&output[(j + 0) * rows + i], tt0);
            _mm256_storeu_ps(&output[(j + 1) * rows + i], tt1);
            _mm256_storeu_ps(&output[(j + 2) * rows + i], tt2);
            _mm256_storeu_ps(&output[(j + 3) * rows + i], tt3);
            _mm256_storeu_ps(&output[(j + 4) * rows + i], tt4);
            _mm256_storeu_ps(&output[(j + 5) * rows + i], tt5);
            _mm256_storeu_ps(&output[(j + 6) * rows + i], tt6);
            _mm256_storeu_ps(&output[(j + 7) * rows + i], tt7);
        }
    }

    // Handle remaining elements if dimensions aren't multiples of 8
    // (This shouldn't be reached with our early check, but included for completeness)
    for (int i = (rows / 8) * 8; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }

    for (int i = 0; i < rows; i++) {
        for (int j = (cols / 8) * 8; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}