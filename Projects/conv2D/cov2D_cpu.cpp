#include "conv2D.h"
#include <algorithm> // For std::max/min
#include <immintrin.h> // For AVX2 instructions
#include <vector>     // For remainder loop (optional but clean)
#include <cstring>    // For std::memset
#include <cstdint>    // For int16_t
using namespace std;
#include <iostream>


// Helper function: Applies AVX convolution to a single grayscale plane
static void convolve_plane_avx(
    const unsigned char* input_plane, // Input grayscale plane (padded)
    const unsigned char* kernel,      // Grayscale kernel
    unsigned char* output_plane,      // Output grayscale plane (original size)
    int original_rows, int original_cols,
    int kernel_rows, int kernel_cols,
    int padded_cols                   // Width of the padded input plane
) {
    int pad = (kernel_rows - 1) / 2; // Assuming square kernel & padding

    // Calculate kernel sum for proper normalization
    float kernel_sum = 0.0f;
    for (int k = 0; k < kernel_rows * kernel_cols; k++) {
        kernel_sum += kernel[k] / 255.0f; 
    }
    
    // Debug print kernel values and sum
    std::cout << "Kernel sum: " << kernel_sum << std::endl;
    std::cout << "Kernel values: ";
    for (int i = 0; i < min(5, kernel_rows * kernel_cols); i++) {
        std::cout << (int)kernel[i] << " ";
    }
    std::cout << std::endl;
    
    // For identity kernels or when sum is too small, use special handling
    if (kernel_sum < 0.9f) {
        std::cout << "Using special normalization for small kernel sum" << std::endl;
        // Identity kernel or similar - don't normalize by sum
        kernel_sum = 1.0f;
    }
    
    // Scale factor for fixed-point math (q15 format)
    int16_t norm_factor = static_cast<int16_t>((32767.0f / kernel_sum) + 0.5f);
    
    // Debug print normalization factor
    std::cout << "Normalization factor: " << norm_factor << std::endl;
    
    __m256i vec_norm_factor = _mm256_set1_epi16(norm_factor);

    const int step = 16;
    int vec_cols = original_cols - (original_cols % step);

    for (int i = 0; i < original_rows; ++i) { // Output row
        int j = 0;
        for (; j < vec_cols; j += step) { // Output col (vectorized)
            __m256i vec_accum = _mm256_setzero_si256(); // 16-bit accumulator

            for (int ky = 0; ky < kernel_rows; ++ky) {
                for (int kx = 0; kx < kernel_cols; ++kx) {
                    // Calculate input position - explicitly show offset calculation
                    const unsigned char* input_ptr = input_plane + (i + ky) * padded_cols + (j + kx);
                    
                    // Load 16 input pixels
                    __m128i input_bytes = _mm_loadu_si128((__m128i const*)input_ptr);
                    __m256i input_epi16 = _mm256_cvtepu8_epi16(input_bytes);
                    
                    // Get kernel value
                    unsigned char kernel_byte = kernel[ky * kernel_cols + kx];
                    __m256i kernel_epi16 = _mm256_set1_epi16(static_cast<int16_t>(kernel_byte));
                    
                    // Multiply and accumulate
                    __m256i prod_epi16 = _mm256_mullo_epi16(input_epi16, kernel_epi16);
                    vec_accum = _mm256_add_epi16(vec_accum, prod_epi16);
                }
            }

            // Normalize using high-precision rounding signed multiply
            __m256i norm_result_epi16 = _mm256_mulhrs_epi16(vec_accum, vec_norm_factor);

            // Clamp to ensure values stay in 0-255 range (extra safety)
            // This isn't strictly necessary with packus, but adds clarity
            __m256i zero = _mm256_setzero_si256();
            __m256i max_val = _mm256_set1_epi16(255);
            norm_result_epi16 = _mm256_max_epi16(_mm256_min_epi16(norm_result_epi16, max_val), zero);

            // Pack and Store
            __m128i low128 = _mm256_castsi256_si128(norm_result_epi16);
            __m128i high128 = _mm256_extracti128_si256(norm_result_epi16, 1);
            __m128i final_bytes = _mm_packus_epi16(low128, high128);
            _mm_storeu_si128((__m128i*)(output_plane + i * original_cols + j), final_bytes);
        }

        // Handle Remainder Pixels (Scalar) - Match the exact same normalization logic
        for (; j < original_cols; ++j) {
            float sum = 0.0f;
            for (int ky = 0; ky < kernel_rows; ++ky) {
                for (int kx = 0; kx < kernel_cols; ++kx) {
                    int in_y = i + ky;
                    int in_x = j + kx;
                    int in_idx = in_y * padded_cols + in_x;
                    float k_val = kernel[ky * kernel_cols + kx] / 255.0f;
                    sum += input_plane[in_idx] * k_val;
                }
            }
            
            // Apply the same normalization logic for consistency
            if (kernel_sum < 0.9f) {
                // Identity kernel special case - don't divide
            } else {
                sum = sum / kernel_sum;
            }
            
            output_plane[i * original_cols + j] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, sum)));
        }
    }

    // Debug: Print the first few pixels of the output for verification
    std::cout << "First 5 output pixels: ";
    for (int i = 0; i < 5 && i < original_cols; i++) {
        std::cout << (int)output_plane[i] << " ";
    }
    std::cout << std::endl;
}

// Original scalar implementation (keep for reference/fallback)
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

// AVX implementation for 3 Channels using plane separation
void conv2D_cpu_avx(unsigned char* padded_input, unsigned char* kernel, unsigned char* output,
                   int original_rows, int original_cols, int kernel_rows, int kernel_cols, int channels)
{
    // Print the kernel type detection
    bool is_identity = true;
    bool is_box_blur = true;
    unsigned char first_non_center = 0;
    
    // Check if it's an identity kernel (center = 255, rest = 0)
    int center_idx = (kernel_rows/2) * kernel_cols + (kernel_cols/2);
    for (int i = 0; i < kernel_rows * kernel_cols; i++) {
        if (i == center_idx) {
            if (kernel[i] != 255) is_identity = false;
        } else {
            if (kernel[i] != 0) is_identity = false;
            if (first_non_center == 0 && kernel[i] != 0) first_non_center = kernel[i];
        }
        
        // Check if all values are the same (box blur)
        if (i > 0 && kernel[i] != kernel[0]) is_box_blur = false;
    }
    
    std::cout << "Kernel type: " 
              << (is_identity ? "Identity" : is_box_blur ? "Box blur" : "Custom") 
              << ", First non-center: " << (int)first_non_center 
              << ", Center: " << (int)kernel[center_idx] << std::endl;
                
    if (channels != 3) {
        // Fallback for non-3-channel images
        fprintf(stderr, "Warning: AVX implementation requires 3 channels. Falling back to scalar.\n");
        conv2D_cpu(padded_input, kernel, output, original_rows, original_cols, kernel_rows, kernel_cols, channels);
        return;
    }

    int pad = (kernel_rows - 1) / 2;
    int padded_width = original_cols + 2 * pad;
    int padded_height = original_rows + 2 * pad;
    size_t plane_padded_size = static_cast<size_t>(padded_width) * padded_height;
    size_t plane_original_size = static_cast<size_t>(original_cols) * original_rows;

    // --- Allocate temporary plane buffers ---
    unsigned char* plane_b_in = new unsigned char[plane_padded_size];
    unsigned char* plane_g_in = new unsigned char[plane_padded_size];
    unsigned char* plane_r_in = new unsigned char[plane_padded_size];

    unsigned char* plane_b_out = new unsigned char[plane_original_size];
    unsigned char* plane_g_out = new unsigned char[plane_original_size];
    unsigned char* plane_r_out = new unsigned char[plane_original_size];
    // --- End Allocation ---

    // --- De-interleave BGR input into separate planes ---
    for (int r = 0; r < padded_height; ++r) {
        for (int col = 0; col < padded_width; ++col) {
            int plane_idx = r * padded_width + col;
            int interleaved_idx = plane_idx * channels;
            plane_b_in[plane_idx] = padded_input[interleaved_idx + 0]; // B
            plane_g_in[plane_idx] = padded_input[interleaved_idx + 1]; // G
            plane_r_in[plane_idx] = padded_input[interleaved_idx + 2]; // R
        }
    }
    // --- End De-interleave ---

    // --- Convolve each plane using the AVX helper ---
    convolve_plane_avx(plane_b_in, kernel, plane_b_out, original_rows, original_cols, kernel_rows, kernel_cols, padded_width);
    convolve_plane_avx(plane_g_in, kernel, plane_g_out, original_rows, original_cols, kernel_rows, kernel_cols, padded_width);
    convolve_plane_avx(plane_r_in, kernel, plane_r_out, original_rows, original_cols, kernel_rows, kernel_cols, padded_width);
    // --- End Convolution ---

    // --- Interleave separate output planes back into the final BGR output ---
    for (int r = 0; r < original_rows; ++r) {
        for (int col = 0; col < original_cols; ++col) {
            int plane_idx = r * original_cols + col;
            int interleaved_idx = plane_idx * channels;
            output[interleaved_idx + 0] = plane_b_out[plane_idx]; // B
            output[interleaved_idx + 1] = plane_g_out[plane_idx]; // G
            output[interleaved_idx + 2] = plane_r_out[plane_idx]; // R
        }
    }
    // --- End Interleave ---

    // --- Cleanup temporary buffers ---
    delete[] plane_b_in;
    delete[] plane_g_in;
    delete[] plane_r_in;
    delete[] plane_b_out;
    delete[] plane_g_out;
    delete[] plane_r_out;
    // --- End Cleanup ---
}