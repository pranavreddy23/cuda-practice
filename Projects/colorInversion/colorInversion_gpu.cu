#include <cuda_runtime.h>
#include "../common/profiler.h"
#include "colorInversion.h"
#include <iostream>
#include <float.h>

// Helper function for checking CUDA errors
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the index is within the bounds of the image
    if (idx < width * height) {
        // Each pixel has 4 components: R, G, B, A
        int pixelIndex = idx * 4;

        // Invert the R, G, B components (subtract from 255)
        image[pixelIndex] = 255 - image[pixelIndex];       // Invert Red
        image[pixelIndex + 1] = 255 - image[pixelIndex + 1]; // Invert Green
        image[pixelIndex + 2] = 255 - image[pixelIndex + 2]; // Invert Blue
        // Alpha remains unchanged, so we do not modify image[pixelIndex + 3]
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void colorInversion_gpu(unsigned char* image, int width, int height) {
    static Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    Profiler::GPUTimer total_timer;
    Profiler::GPUTimer kernel_timer;
    
    total_timer.start();
    unsigned char *d_image;
    CHECK_CUDA_ERROR(cudaMalloc(&d_image, sizeof(unsigned char) * width * height * 4));
    
    mem_tracker.record_gpu_allocation(sizeof(unsigned char) * width * height * 4);
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_image, image, sizeof(unsigned char) * width * height * 4, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    kernel_timer.start();
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    kernel_timer.stop();
    Profiler::KernelTimeTracker::record_kernel_time("colorInversion", kernel_timer.elapsed_milliseconds());

    CHECK_CUDA_ERROR(cudaMemcpy(image, d_image, sizeof(unsigned char) * width * height * 4, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaFree(d_image));
    total_timer.stop();
    Profiler::KernelTimeTracker::last_total_time = total_timer.elapsed_milliseconds();
}