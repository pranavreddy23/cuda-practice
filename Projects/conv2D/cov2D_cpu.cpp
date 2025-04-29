#include "conv2D.h"
#include <algorithm> // For std::max/min
#include <immintrin.h> // For AVX2 instructions
#include <vector>     // For remainder loop (optional but clean)
#include <cstring>    // For std::memset

// Assumes 'input' points to the START of the PADDED buffer.
// 'output' points to the START of the output buffer (original size).
// width/height refer to the ORIGINAL image dimensions.
void conv2D_cpu(unsigned char* padded_input, unsigned char* kernel, unsigned char* output,
              int original_rows, int original_cols, int kernel_rows, int kernel_cols, int channels) {
    
    int pad_rows = (kernel_rows - 1) / 2;
    int pad_cols = (kernel_cols - 1) / 2;
    // Width of the PADDED input buffer in pixels
    int padded_width = original_cols + 2 * pad_cols; 
    
    // For each pixel in the output image (original dimensions)
    for (int out_y = 0; out_y < original_rows; out_y++) {
        for (int out_x = 0; out_x < original_cols; out_x++) {
            // For each color channel
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f; // Accumulate as float
                
                // Apply kernel
                for (int ky = 0; ky < kernel_rows; ky++) {
                    for (int kx = 0; kx < kernel_cols; kx++) {
                        // Calculate the corresponding pixel coordinates in the PADDED input buffer.
                        // The top-left corner of the kernel overlay corresponds to (out_x, out_y) 
                        // in the padded buffer's coordinate system (relative to its top-left).
                        int in_y = out_y + ky; 
                        int in_x = out_x + kx; 
                        
                        // Calculate the 1D index for the padded input pixel
                        // (Row * Width + Column) * NumChannels + ChannelOffset
                        int in_idx = (in_y * padded_width + in_x) * channels + c;
                        
                        // Get kernel value (normalized from 0-255 back to 0-1 range)
                        // CORRECT KERNEL INDEXING: row * num_cols + col
                        float k_val = kernel[ky * kernel_cols + kx] / 255.0f; 
                        
                        // Accumulate the weighted input value
                        sum += padded_input[in_idx] * k_val; 
                    }
                }
                
                // Calculate the 1D index for the output pixel
                int out_idx = (out_y * original_cols + out_x) * channels + c;
                
                // Write the final clamped value to the output buffer
                output[out_idx] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, sum))); 
            }
        }
    }
}

void conv2D_cpu_avx(unsigned char* input, unsigned char* kernel, unsigned char* output,
                   int original_rows, int original_cols, int kernel_rows, int kernel_cols, int channels) {
    // Padding size
    int pad_rows = (kernel_rows - 1) / 2;
    int pad_cols = (kernel_cols - 1) / 2;

    // Dimensions of the padded input buffer
    int padded_rows = original_rows + 2 * pad_rows;
    int padded_cols = original_cols + 2 * pad_cols;
    
    // Process each channel separately
    for (int c = 0; c < channels; c++) {
        // For each row in output
        for (int i = 0; i < original_rows; i++) {
            int j = 0;
            
            // Vector processing - process 16 pixels at a time
            for (; j <= original_cols - 16; j += 16) {
                // Initialize accumulators for 16 output pixels
                __m256i accum_low = _mm256_setzero_si256();   // For pixels j to j+7
                __m256i accum_high = _mm256_setzero_si256();  // For pixels j+8 to j+15
                
                // Apply kernel
                for (int kr = 0; kr < kernel_rows; kr++) {
                    for (int kc = 0; kc < kernel_cols; kc++) {
                        // Get kernel value at position (kr, kc)
                        unsigned char kernel_val = kernel[kr * kernel_cols + kc];
                        // Skip zero kernel values for efficiency
                        if (kernel_val == 0) continue;
                        
                        // Calculate input row and column with padding offset
                        int in_row = i + kr;
                        int in_col = j + kc;
                        
                        // Input pointer for current position and channel
                        const unsigned char* in_ptr = &input[(in_row * padded_cols + in_col) * channels + c];
                        
                        // Load 16 input values for this channel (with stride of 'channels')
                        unsigned char temp_buffer[16];
                        for (int p = 0; p < 16; p++) {
                            temp_buffer[p] = in_ptr[p * channels];
                        }
                        
                        // Load gathered values
                        __m128i input_bytes = _mm_loadu_si128((__m128i*)temp_buffer);
                        
                        // Broadcast kernel value to 8x 16-bit lanes
                        __m128i kernel_epi16 = _mm_set1_epi16(kernel_val);
                        
                        // Process first 8 pixels
                        __m128i input_epi16_low = _mm_cvtepu8_epi16(input_bytes);
                        __m128i prod_epi16_low = _mm_mullo_epi16(input_epi16_low, kernel_epi16);
                        __m256i prod_epi32_low = _mm256_cvtepi16_epi32(prod_epi16_low);
                        accum_low = _mm256_add_epi32(accum_low, prod_epi32_low);
                        
                        // Process second 8 pixels
                        __m128i input_bytes_high = _mm_unpackhi_epi64(input_bytes, _mm_setzero_si128());
                        __m128i input_epi16_high = _mm_cvtepu8_epi16(input_bytes_high);
                        __m128i prod_epi16_high = _mm_mullo_epi16(input_epi16_high, kernel_epi16);
                        __m256i prod_epi32_high = _mm256_cvtepi16_epi32(prod_epi16_high);
                        accum_high = _mm256_add_epi32(accum_high, prod_epi32_high);
                    }
                }
                
                // Pack results to 16-bit (saturate to range -32768 to 32767)
                __m128i packed16_low = _mm_packs_epi32(
                    _mm256_castsi256_si128(accum_low),
                    _mm256_extracti128_si256(accum_low, 1)
                );
                __m128i packed16_high = _mm_packs_epi32(
                    _mm256_castsi256_si128(accum_high),
                    _mm256_extracti128_si256(accum_high, 1)
                );
                
                // Pack results to 8-bit (saturate to range 0 to 255)
                __m128i result_bytes = _mm_packus_epi16(packed16_low, packed16_high);
                
                // Store results with channel stride
                unsigned char temp_output[16];
                _mm_storeu_si128((__m128i*)temp_output, result_bytes);
                
                // Write back to output with proper channel interleaving
                for (int p = 0; p < 16; p++) {
                    output[((i * original_cols) + j + p) * channels + c] = temp_output[p];
                }
            }
            
            // Handle remaining pixels with scalar code
            for (; j < original_cols; j++) {
                int sum = 0;
                
                // Apply kernel
                for (int kr = 0; kr < kernel_rows; kr++) {
                    for (int kc = 0; kc < kernel_cols; kc++) {
                        int in_row = i + kr;
                        int in_col = j + kc;
                        
                        // Get input value with channel offset
                        unsigned char in_val = input[(in_row * padded_cols + in_col) * channels + c];
                        
                        // Get kernel value
                        unsigned char k_val = kernel[kr * kernel_cols + kc];
                        
                        // Accumulate
                        sum += in_val * k_val / 255;
                    }
                }
                
                // Clamp to [0, 255] and store
                output[(i * original_cols + j) * channels + c] = 
                    static_cast<unsigned char>(std::max(0, std::min(255, sum)));
            }
        }
    }
}