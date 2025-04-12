#include "reverseArray.h"
#include <immintrin.h> // For AVX2 intrinsics

void reverseArray_avx(float* input, int N) {
    // Process 8 elements at a time using AVX2
    const int simd_width = 8; // AVX2 processes 8 floats at once
    
    // Only use SIMD for arrays large enough
    if (N >= simd_width * 2) {
        int i = 0;
        int j = N - simd_width;
        
        // Process elements in chunks of 8 from both ends
        while (i < j) {
            // Load 8 elements from the beginning
            __m256 vec_start = _mm256_loadu_ps(&input[i]);
            
            // Load 8 elements from the end
            __m256 vec_end = _mm256_loadu_ps(&input[j]);
            
            // Reverse the vectors
            vec_start = _mm256_permute2f128_ps(vec_start, vec_start, 0x01); // Swap 128-bit lanes
            vec_start = _mm256_shuffle_ps(vec_start, vec_start, 0x1B);      // Reverse within 128-bit lanes
            
            vec_end = _mm256_permute2f128_ps(vec_end, vec_end, 0x01);       // Swap 128-bit lanes
            vec_end = _mm256_shuffle_ps(vec_end, vec_end, 0x1B);            // Reverse within 128-bit lanes
            
            // Store the swapped vectors
            _mm256_storeu_ps(&input[j], vec_start);
            _mm256_storeu_ps(&input[i], vec_end);
            
            i += simd_width;
            j -= simd_width;
        }
        
        // Handle remaining elements in the middle (if any)
        i = (N / 2) - ((N / 2) % simd_width);
        j = N - i - 1;
        
        while (i < j) {
            float temp = input[i];
            input[i] = input[j];
            input[j] = temp;
            i++;
            j--;
        }
    } else {
        // Fall back to scalar implementation for small arrays
        for (int i = 0; i < N / 2; i++) {
            float temp = input[i];
            input[i] = input[N - i - 1];
            input[N - i - 1] = temp;
        }
    }
} 