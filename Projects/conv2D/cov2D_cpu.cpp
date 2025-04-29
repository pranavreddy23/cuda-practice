#include "conv2D.h"
#include <algorithm> // For std::max/min
#include <immintrin.h> // For AVX2 instructions
#include <vector>     // For remainder loop (optional but clean)

// Assumes 'input' is already padded, and 'output' has dimensions of the original image.
void conv2D_cpu(unsigned char* input, unsigned char* kernel, unsigned char* output,
               int original_rows, int original_cols, int kernel_rows, int kernel_cols) {

    // Padding size (assuming kernel dimensions are odd)
    int pad_rows = (kernel_rows - 1) / 2;
    int pad_cols = (kernel_cols - 1) / 2;

    // Dimensions of the padded input buffer
    int padded_rows = original_rows + 2 * pad_rows;
    int padded_cols = original_cols + 2 * pad_cols;

    // Iterate over each pixel of the *original* image (which is the size of the output)
    for (int i = 0; i < original_rows; i++) {
        for (int j = 0; j < original_cols; j++) {
            int sum = 0;
            // Apply the kernel
            for (int k = 0; k < kernel_rows; k++) {
                for (int l = 0; l < kernel_cols; l++) {
                    // Calculate corresponding coordinates in the *padded* input
                    int input_row = i + k; // Index relative to top-left of padded input
                    int input_col = j + l;

                    // Access padded input
                    sum += input[input_row * padded_cols + input_col] * kernel[k * kernel_cols + l];
                }
            }
            // Write result to the output buffer (sized as the original image)
            // Clamp the result to [0, 255] for unsigned char
            output[i * original_cols + j] = static_cast<unsigned char>(std::max(0, std::min(255, sum)));
        }
    }
}

void conv2D_cpu_avx(unsigned char* input, unsigned char* kernel, unsigned char* output,
                   int original_rows, int original_cols, int kernel_rows, int kernel_cols) {

    // Padding size
    int pad_rows = (kernel_rows - 1) / 2;
    int pad_cols = (kernel_cols - 1) / 2;

    // Dimensions of the padded input buffer
    int padded_rows = original_rows + 2 * pad_rows;
    int padded_cols = original_cols + 2 * pad_cols;

    // Process 16 pixels at a time using AVX2 (operating on 128-bit lanes mostly)
    const int step = 16;
    int vec_cols = original_cols - (original_cols % step);

    for (int i = 0; i < original_rows; ++i) {
        int j = 0;
        for (; j < vec_cols; j += step) {
            // Accumulators for 16 pixels (using two 256-bit vectors for 8x epi32 each)
            __m256i accum_low = _mm256_setzero_si256();  // Accumulators for pixels j to j+7
            __m256i accum_high = _mm256_setzero_si256(); // Accumulators for pixels j+8 to j+15

            // Apply the kernel
            for (int k = 0; k < kernel_rows; ++k) {
                for (int l = 0; l < kernel_cols; ++l) {
                    // Load 16 input bytes (pixels) from the padded input
                    const unsigned char* input_ptr = input + (i + k) * padded_cols + (j + l);
                    __m128i input_bytes = _mm_loadu_si128((__m128i const*)input_ptr);

                    // Load the single kernel byte for this position
                    unsigned char kernel_byte = kernel[k * kernel_cols + l];

                    // --- Process Pixels j to j+7 ---
                    // Unpack 8 input bytes to 8 16-bit integers
                    __m128i input_epi16_low = _mm_cvtepu8_epi16(input_bytes);
                    // Broadcast kernel byte to 8 16-bit integers
                    __m128i kernel_epi16_low = _mm_set1_epi16(kernel_byte);
                    // Multiply input and kernel (epi16)
                    __m128i prod_epi16_low = _mm_mullo_epi16(input_epi16_low, kernel_epi16_low);
                    // Unpack product to 32-bit integers
                    __m256i prod_epi32_low = _mm256_cvtepi16_epi32(prod_epi16_low);
                    // Add to 32-bit accumulator
                    accum_low = _mm256_add_epi32(accum_low, prod_epi32_low);

                    // --- Process Pixels j+8 to j+15 ---
                    // Extract upper 8 bytes from input_bytes (requires shuffle/cast or separate load)
                     // A simpler way is to unpack the upper half directly if possible, or just load again slightly offset
                     // Let's unpack the upper part from the original 128 bit load
                    __m128i input_bytes_high_half = _mm_unpackhi_epi64(input_bytes, _mm_setzero_si128()); // Get upper 64 bits into lower part of a new vector
                    __m128i input_epi16_high = _mm_cvtepu8_epi16(input_bytes_high_half); // Unpack the 8 bytes from the upper half
                    // Broadcast kernel byte (already done, can reuse kernel_epi16_low or make a new one)
                    __m128i kernel_epi16_high = kernel_epi16_low; // Reuse broadcasted kernel value
                    // Multiply
                    __m128i prod_epi16_high = _mm_mullo_epi16(input_epi16_high, kernel_epi16_high);
                    // Unpack product to 32-bit integers
                    __m256i prod_epi32_high = _mm256_cvtepi16_epi32(prod_epi16_high);
                    // Add to 32-bit accumulator
                    accum_high = _mm256_add_epi32(accum_high, prod_epi32_high);
                }
            }

            // --- Pack and Store Results ---
            // Clamp 32-bit results to 16-bit signed (-32768 to 32767)
            // _mm_packs_epi32 saturates. Since our values should be positive, this works like clamping max to 32767.
            __m128i packed16_low  = _mm_packs_epi32(_mm256_castsi256_si128(accum_low), _mm256_extracti128_si256(accum_low, 1));
            __m128i packed16_high = _mm_packs_epi32(_mm256_castsi256_si128(accum_high), _mm256_extracti128_si256(accum_high, 1));

            // Clamp 16-bit results to 8-bit unsigned (0 to 255)
            // _mm_packus_epi16 saturates.
            __m128i final_bytes = _mm_packus_epi16(packed16_low, packed16_high);

            // Store the 16 resulting bytes
            _mm_storeu_si128((__m128i*)(output + i * original_cols + j), final_bytes);
        }

        // --- Handle Remainder Pixels (Scalar) ---
        for (; j < original_cols; ++j) {
            int sum = 0;
            for (int k = 0; k < kernel_rows; ++k) {
                for (int l = 0; l < kernel_cols; ++l) {
                    int input_row = i + k;
                    int input_col = j + l;
                    sum += input[input_row * padded_cols + input_col] * kernel[k * kernel_cols + l];
                }
            }
            output[i * original_cols + j] = static_cast<unsigned char>(std::max(0, std::min(255, sum)));
        }
    }
}

