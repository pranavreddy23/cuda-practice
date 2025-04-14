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

// Performance comparison helper for multiple approaches
class MultiApproachComparison {
private:
    std::string test_name;
    double baseline_time_ms;  // Usually the linear CPU implementation time
    std::map<std::string, double> approach_times;
    std::map<std::string, bool> approach_verified;
    
public:
    MultiApproachComparison(const std::string& name) 
        : test_name(name), baseline_time_ms(0) {}
    
    void set_baseline_time(double time_ms) {
        baseline_time_ms = time_ms;
    }
    
    void record_approach_time(const std::string& approach_name, double time_ms) {
        approach_times[approach_name] = time_ms;
    }
    
    void set_approach_verified(const std::string& approach_name, bool status) {
        approach_verified[approach_name] = status;
    }
    
    double get_approach_time(const std::string& approach_name) const {
        auto it = approach_times.find(approach_name);
        if (it != approach_times.end()) {
            return it->second;
        }
        return 0.0;
    }
    
    double get_speedup(const std::string& approach_name) const {
        auto it = approach_times.find(approach_name);
        if (it != approach_times.end() && baseline_time_ms > 0 && it->second > 0) {
            return baseline_time_ms / it->second;
        }
        return 0.0;
    }
    
    std::vector<std::string> get_approach_names() const {
        std::vector<std::string> names;
        for (const auto& pair : approach_times) {
            names.push_back(pair.first);
        }
        return names;
    }
    
    void print_summary() const {
        std::cout << "\n===== " << test_name << " Performance Summary =====" << std::endl;
        
        // Print baseline time
        std::cout << "Baseline (CPU Linear): " << baseline_time_ms << " ms" << std::endl;
        
        // Print all approach times and speedups
        for (const auto& pair : approach_times) {
            const std::string& name = pair.first;
            double time = pair.second;
            double speedup = (baseline_time_ms > 0 && time > 0) ? (baseline_time_ms / time) : 0.0;
            
            std::cout << name << " Time: " << time << " ms";
            if (speedup > 0) {
                std::cout << " (Speedup: " << speedup << "x)";
            }
            
            auto verify_it = approach_verified.find(name);
            if (verify_it != approach_verified.end()) {
                std::cout << " - Verification: " << (verify_it->second ? "PASSED" : "FAILED");
            }
            
            std::cout << std::endl;
        }
    }
};

// GPU kernel time tracker for multiple kernels
class KernelTimeTracker {
private:
    static std::map<std::string, std::vector<float>> kernel_times;
    
public:
    static float last_total_time;
    
    // Reset all timings
    static void reset() {
        last_total_time = 0.0f;
        kernel_times.clear();
    }
    
    // Record a kernel's execution time
    static void record_kernel_time(const std::string& kernel_name, float time_ms) {
        kernel_times[kernel_name].push_back(time_ms);
        // Don't update last_total_time here - let the caller handle that
    }
    
    // Get total time for a specific kernel
    static float get_total_kernel_time(const std::string& kernel_name) {
        float total = 0.0f;
        auto it = kernel_times.find(kernel_name);
        if (it != kernel_times.end()) {
            for (float time : it->second) {
                total += time;
            }
        }
        return total;
    }
    
    // Print a summary of all kernel times
    static void print_kernel_times() {
        std::cout << "GPU Kernel Timing Breakdown:" << std::endl;
        
        for (const auto& pair : kernel_times) {
            const std::string& kernel_name = pair.first;
            float total_time = 0.0f;
            for (float time : pair.second) {
                total_time += time;
            }
            std::cout << "  " << kernel_name << ": " << total_time << " ms" << std::endl;
        }
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