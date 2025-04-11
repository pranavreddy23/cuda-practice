#include "reverseArray.h"
#include <cuda_runtime.h>
#include "../common/profiler.h"
#define TILE_SIZE 256
__global__ void reverse_array(float* input, int N) {
    
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int reverse = N -1- gid;

    if ( gid < N /2){
        float tmp = input[gid];
        input[gid] = input[reverse];
        input[reverse] = tmp;
    }


}

// input is device pointer
void reverseArray_gpu(float* input, int N) {
    static Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    Profiler::GPUTimer total_timer;
    Profiler::GPUTimer kernel_timer;

    total_timer.start();
    float *d_input;
    cudaMalloc(&d_input, sizeof(float) * N);
    
    mem_tracker.record_gpu_allocation(sizeof(float) * N);

    cudaMemcpy(d_input, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernel_timer.start();
    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    kernel_timer.stop();
    Profiler::KernelTimeTracker::record_kernel_time("reverseArray", kernel_timer.elapsed_milliseconds());

    cudaMemcpy(input, d_input, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaDeviceSynchronize();
    total_timer.stop();
    Profiler::KernelTimeTracker::last_total_time = total_timer.elapsed_milliseconds();
}