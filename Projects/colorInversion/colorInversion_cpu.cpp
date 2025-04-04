#include "colorInversion.h"
#include <immintrin.h> // For AVX/AVX2 intrinsics
#include <cstring>     // For memcpy

// Non-vectorized fallback implementation
void colorInversion_cpu_scalar(unsigned char* image, int width, int height) {
    int totalPixels = width * height;
    
    for (int i = 0; i < totalPixels; i++) {
        int pixelIndex = i * 4;
        
        // Invert R, G, B components (subtract from 255)
        image[pixelIndex] = 255 - image[pixelIndex];         // R
        image[pixelIndex + 1] = 255 - image[pixelIndex + 1]; // G
        image[pixelIndex + 2] = 255 - image[pixelIndex + 2]; // B
        // Alpha remains unchanged (image[pixelIndex + 3])
    }
}

// AVX2 optimized implementation
void colorInversion_cpu_avx2(unsigned char* image, int width, int height) {
    int totalPixels = width * height;
    int totalBytes = totalPixels * 4;
    
    // Process 32 bytes (8 pixels) at a time with AVX2
    int vectorizedLimit = totalBytes - (totalBytes % 32);
    
    // Create a vector of all 255s for the inversion operation
    __m256i allOnes = _mm256_set1_epi8(255);
    
    // Create a mask for the alpha channel (every 4th byte)
    // 0xFF for alpha (to keep original), 0x00 for RGB (to invert)
    __m256i alphaMask = _mm256_set_epi8(
        0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00,
        0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00,
        0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00,
        0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00
    );
    
    // Process 32 bytes (8 pixels) at a time
    for (int i = 0; i < vectorizedLimit; i += 32) {
        // Load 32 bytes (8 pixels) from the image
        __m256i pixels = _mm256_loadu_si256((__m256i*)&image[i]);
        
        // Invert all bytes
        __m256i inverted = _mm256_sub_epi8(allOnes, pixels);
        
        // Blend the inverted pixels with the original alpha values
        // This keeps alpha unchanged while inverting R,G,B
        __m256i result = _mm256_blendv_epi8(inverted, pixels, alphaMask);
        
        // Store the result back to memory
        _mm256_storeu_si256((__m256i*)&image[i], result);
    }
    
    // Handle remaining pixels with scalar code
    for (int i = vectorizedLimit; i < totalBytes; i += 4) {
        image[i] = 255 - image[i];         // R
        image[i + 1] = 255 - image[i + 1]; // G
        image[i + 2] = 255 - image[i + 2]; // B
        // Alpha remains unchanged (image[i + 3])
    }
}

// SSE4.1 optimized implementation (for systems without AVX2)
void colorInversion_cpu_sse(unsigned char* image, int width, int height) {
    int totalPixels = width * height;
    int totalBytes = totalPixels * 4;
    
    // Process 16 bytes (4 pixels) at a time with SSE
    int vectorizedLimit = totalBytes - (totalBytes % 16);
    
    // Create a vector of all 255s for the inversion operation
    __m128i allOnes = _mm_set1_epi8(255);
    
    // Create a mask for the alpha channel (every 4th byte)
    // 0xFF for alpha (to keep original), 0x00 for RGB (to invert)
    __m128i alphaMask = _mm_set_epi8(
        0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00,
        0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00
    );
    
    // Process 16 bytes (4 pixels) at a time
    for (int i = 0; i < vectorizedLimit; i += 16) {
        // Load 16 bytes (4 pixels) from the image
        __m128i pixels = _mm_loadu_si128((__m128i*)&image[i]);
        
        // Invert all bytes
        __m128i inverted = _mm_sub_epi8(allOnes, pixels);
        
        // Blend the inverted pixels with the original alpha values
        // This keeps alpha unchanged while inverting R,G,B
        __m128i result = _mm_blendv_epi8(inverted, pixels, alphaMask);
        
        // Store the result back to memory
        _mm_storeu_si128((__m128i*)&image[i], result);
    }
    
    // Handle remaining pixels with scalar code
    for (int i = vectorizedLimit; i < totalBytes; i += 4) {
        image[i] = 255 - image[i];         // R
        image[i + 1] = 255 - image[i + 1]; // G
        image[i + 2] = 255 - image[i + 2]; // B
        // Alpha remains unchanged (image[i + 3])
    }
}

// Main CPU implementation that selects the best available method
void colorInversion_cpu(unsigned char* image, int width, int height) {
    // Check CPU support for AVX2 or SSE4.1 at runtime
    // This is a simplified approach - in production code, you might want to use
    // CPU feature detection libraries or compiler intrinsics
    
    #if defined(__AVX2__)
        // Use AVX2 implementation if supported
        colorInversion_cpu_avx2(image, width, height);
    #elif defined(__SSE4_1__)
        // Fall back to SSE4.1 if AVX2 is not available
        colorInversion_cpu_sse(image, width, height);
    #else
        // Fall back to scalar implementation if neither is available
        colorInversion_cpu_scalar(image, width, height);
    #endif
}