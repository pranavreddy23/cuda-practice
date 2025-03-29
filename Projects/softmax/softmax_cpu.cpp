#include "softmax.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <float.h>

void softmax_cpu(const float* input, float* output, int N) {
    // Step 1: Find the maximum value (for numerical stability)
    float max_val = -FLT_MAX;
    
    // SIMD version of finding maximum
    if (N >= 8) {
        __m256 max_vec = _mm256_set1_ps(-FLT_MAX);
        int i = 0;
        
        // Process 8 elements at a time
        for (; i <= N - 8; i += 8) {
            __m256 in_vec = _mm256_loadu_ps(&input[i]);
            max_vec = _mm256_max_ps(max_vec, in_vec);
        }
        
        // Find maximum value in the vector
        float max_array[8];
        _mm256_storeu_ps(max_array, max_vec);
        
        for (int j = 0; j < 8; j++) {
            max_val = std::max(max_val, max_array[j]);
        }
        
        // Handle remaining elements
        for (; i < N; i++) {
            max_val = std::max(max_val, input[i]);
        }
    } else {
        // For small arrays, use scalar code
        for (int i = 0; i < N; i++) {
            max_val = std::max(max_val, input[i]);
        }
    }
    
    // Step 2: Compute exp(x - max) for each element and sum them
    float sum = 0.0f;
    
    // SIMD version of computing exponentials and sum
    if (N >= 8) {
        __m256 sum_vec = _mm256_setzero_ps();
        __m256 max_vec = _mm256_set1_ps(max_val);
        int i = 0;
        
        // Process 8 elements at a time
        for (; i <= N - 8; i += 8) {
            // Load input values
            __m256 in_vec = _mm256_loadu_ps(&input[i]);
            
            // Subtract max value for numerical stability
            __m256 shifted_vec = _mm256_sub_ps(in_vec, max_vec);
            
            // Compute exponential using polynomial approximation
            // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120 + x⁶/720
            // This is accurate enough for x in [-0.5, 0.5]
            // For larger ranges, we'll use scalar exp for simplicity
            
            // Store exp results
            float exp_results[8];
            _mm256_storeu_ps(exp_results, shifted_vec);
            
            // Compute exponentials and store back
            for (int j = 0; j < 8; j++) {
                exp_results[j] = std::exp(exp_results[j]);
            }
            
            // Load computed exponentials
            __m256 exp_vec = _mm256_loadu_ps(exp_results);
            
            // Store results for later normalization
            _mm256_storeu_ps(&output[i], exp_vec);
            
            // Accumulate sum
            sum_vec = _mm256_add_ps(sum_vec, exp_vec);
        }
        
        // Reduce sum vector to a single value
        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);
        
        for (int j = 0; j < 8; j++) {
            sum += sum_array[j];
        }
        
        // Handle remaining elements
        for (; i < N; i++) {
            float exp_val = std::exp(input[i] - max_val);
            output[i] = exp_val;
            sum += exp_val;
        }
    } else {
        // For small arrays, use scalar code
        for (int i = 0; i < N; i++) {
            float exp_val = std::exp(input[i] - max_val);
            output[i] = exp_val;
            sum += exp_val;
        }
    }
    
    // Step 3: Normalize by dividing each element by the sum
    float inv_sum = 1.0f / sum;
    
    // SIMD version of normalization
    if (N >= 8) {
        __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
        int i = 0;
        
        // Process 8 elements at a time
        for (; i <= N - 8; i += 8) {
            __m256 out_vec = _mm256_loadu_ps(&output[i]);
            out_vec = _mm256_mul_ps(out_vec, inv_sum_vec);
            _mm256_storeu_ps(&output[i], out_vec);
        }
        
        // Handle remaining elements
        for (; i < N; i++) {
            output[i] *= inv_sum;
        }
    } else {
        // For small arrays, use scalar code
        for (int i = 0; i < N; i++) {
            output[i] *= inv_sum;
        }
    }
}
