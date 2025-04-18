#include "vectorAdd.h"
#include <cuda_runtime.h>
#include "../common/profiler.h"

// CUDA kernel for vector addition
__global__ void vectorAdd_kernel(const int *__restrict a, const int *__restrict b,
                               int *__restrict c, int N) {
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Boundary check
  if (tid < N) c[tid] = a[tid] + b[tid];
}

// GPU implementation wrapper
void vectorAdd_gpu(const std::vector<int>& a, const std::vector<int>& b,
                  std::vector<int>& c, int N) {
  size_t bytes = sizeof(int) * N;

  // Create memory tracker
  static Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();

  // Create GPU timer for total time measurement
  Profiler::GPUTimer total_timer;
  total_timer.start();

  // Allocate memory on the device
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  
  // Track GPU memory allocation
  mem_tracker.record_gpu_allocation(bytes * 3); // d_a, d_b, d_c

  // Copy data from the host to the device (CPU -> GPU)
  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

  // Threads per CTA (1024)
  int NUM_THREADS = 1 << 10;

  // CTAs per Grid
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  // Create GPU timer for kernel timing
  Profiler::GPUTimer kernel_timer;
  
  // Start timing just the kernel execution
  kernel_timer.start();
  
  // Launch the kernel on the GPU
  vectorAdd_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
  
  // Stop timing the kernel execution
  kernel_timer.stop();
  
  // Store the kernel time
  Profiler::KernelTimeTracker::last_kernel_time = kernel_timer.elapsed_milliseconds();

  // Copy sum vector from device to host
  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  
  // Stop timing total GPU operation
  total_timer.stop();
  
  // Store the total GPU time
  Profiler::KernelTimeTracker::last_total_time = total_timer.elapsed_milliseconds();
}