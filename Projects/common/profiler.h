#ifndef PROFILER_H
#define PROFILER_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

namespace Profiler {

// CPU Timer class for measuring execution time on the CPU
class CPUTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running;

public:
    CPUTimer() : running(false) {}

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        running = false;
    }

    double elapsed_milliseconds() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            running ? std::chrono::high_resolution_clock::now() - start_time : end_time - start_time
        );
        return duration.count() / 1000.0;
    }

    double elapsed_seconds() const {
        return elapsed_milliseconds() / 1000.0;
    }
};

// GPU Timer class for measuring execution time on the GPU
class GPUTimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    bool initialized;
    bool running;

public:
    GPUTimer() : initialized(false), running(false) {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        initialized = true;
    }

    ~GPUTimer() {
        if (initialized) {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
        }
    }

    void start() {
        cudaEventRecord(start_event, 0);
        running = true;
    }

    void stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        running = false;
    }

    float elapsed_milliseconds() const {
        float elapsed = 0;
        if (!running && initialized) {
            cudaEventElapsedTime(&elapsed, start_event, stop_event);
        }
        return elapsed;
    }
};

// Flexible kernel time tracker
class KernelTimeTracker {
private:
    static std::map<std::string, float> kernel_times;
    
public:
    static float last_total_time;
    
    // Reset all timings
    static void reset() {
        last_total_time = 0.0f;
        kernel_times.clear();
    }
    
    // Record a kernel's execution time
    static void record_kernel_time(const std::string& kernel_name, float time_ms) {
        kernel_times[kernel_name] = time_ms;
    }
    
    // Get a specific kernel's time
    static float get_kernel_time(const std::string& kernel_name) {
        if (kernel_times.find(kernel_name) != kernel_times.end()) {
            return kernel_times[kernel_name];
        }
        return 0.0f;
    }
    
    // Get all kernel names
    static std::vector<std::string> get_kernel_names() {
        std::vector<std::string> names;
        for (const auto& pair : kernel_times) {
            names.push_back(pair.first);
        }
        return names;
    }
    
    // Get total kernel time (sum of all kernels)
    static float get_total_kernel_time() {
        float total = 0.0f;
        for (const auto& pair : kernel_times) {
            total += pair.second;
        }
        return total;
    }
    
    // Print a summary of all kernel times
    static void print_kernel_times() {
        std::cout << "GPU Kernel Timing Breakdown:" << std::endl;
        float total_kernel_time = 0.0f;
        
        for (const auto& pair : kernel_times) {
            std::cout << "  " << pair.first << ": " << pair.second << " ms" << std::endl;
            total_kernel_time += pair.second;
        }
        
        std::cout << "  Total Kernels: " << total_kernel_time << " ms" << std::endl;
        std::cout << "  Memory Ops: " << (last_total_time - total_kernel_time) << " ms" << std::endl;
    }
};

// Memory usage tracker
class MemoryTracker {
private:
    size_t cpu_bytes_allocated;
    size_t gpu_bytes_allocated;
    
    // Private constructor for singleton pattern
    MemoryTracker() : cpu_bytes_allocated(0), gpu_bytes_allocated(0) {}
    
    // Delete copy constructor and assignment operator
    MemoryTracker(const MemoryTracker&) = delete;
    MemoryTracker& operator=(const MemoryTracker&) = delete;
    
public:
    // Get singleton instance
    static MemoryTracker& getInstance() {
        static MemoryTracker instance;
        return instance;
    }
    
    void record_cpu_allocation(size_t bytes) {
        cpu_bytes_allocated += bytes;
    }
    
    void record_gpu_allocation(size_t bytes) {
        gpu_bytes_allocated += bytes;
    }
    
    size_t get_cpu_bytes() const {
        return cpu_bytes_allocated;
    }
    
    size_t get_gpu_bytes() const {
        return gpu_bytes_allocated;
    }
    
    void reset() {
        cpu_bytes_allocated = 0;
        gpu_bytes_allocated = 0;
    }
    
    void print_summary() const {
        std::cout << "Memory Usage:" << std::endl;
        std::cout << "  CPU: " << (cpu_bytes_allocated / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "  GPU: " << (gpu_bytes_allocated / (1024.0 * 1024.0)) << " MB" << std::endl;
    }
};

// Performance comparison helper
class PerformanceComparison {
private:
    std::string test_name;
    double cpu_time_ms;
    double gpu_time_ms;
    bool verified;
    
public:
    PerformanceComparison(const std::string& name) 
        : test_name(name), cpu_time_ms(0), gpu_time_ms(0), verified(false) {}
    
    void set_cpu_time(double time_ms) {
        cpu_time_ms = time_ms;
    }
    
    void set_gpu_time(double time_ms) {
        gpu_time_ms = time_ms;
    }
    
    void set_verified(bool status) {
        verified = status;
    }
    
    double get_speedup() const {
        if (cpu_time_ms > 0 && gpu_time_ms > 0) {
            return cpu_time_ms / gpu_time_ms;
        }
        return 0.0;
    }
    
    void print_summary() const {
        std::cout << "\n===== " << test_name << " Performance Summary =====" << std::endl;
        std::cout << "CPU Time: " << cpu_time_ms << " ms" << std::endl;
        std::cout << "GPU Time (Total): " << gpu_time_ms << " ms" << std::endl;
        
        float total_kernel_time = KernelTimeTracker::get_total_kernel_time();
        std::cout << "GPU Kernel Time: " << total_kernel_time << " ms" << std::endl;
        std::cout << "GPU Data Transfer Time: " << (gpu_time_ms - total_kernel_time) << " ms" << std::endl;
        
        double speedup = get_speedup();
        if (speedup > 0) {
            std::cout << "Speedup (Total): " << speedup << "x" << std::endl;
            
            if (total_kernel_time > 0) {
                double kernel_speedup = cpu_time_ms / total_kernel_time;
                std::cout << "Speedup (Kernel Only): " << kernel_speedup << "x" << std::endl;
            }
        } else {
            std::cout << "Speedup: N/A" << std::endl;
        }
        
        std::cout << "Verification: " << (verified ? "PASSED" : "FAILED") << std::endl;
    }
};

// CUDA error checking helper
inline void check_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) 
                  << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), __FILE__, __LINE__)

// Device properties reporter
inline void print_device_properties() {
    int device_count;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    
    std::cout << "\n===== CUDA Device Information =====" << std::endl;
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
    }
}

} // namespace Profiler

#endif // PROFILER_H