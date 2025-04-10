#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "../common/profiler.h"
#include "conv1d.h"

// Function to verify results
bool verify_results(const float* cpu_output, const float* gpu_output, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; i++) {
        if (std::abs(cpu_output[i] - gpu_output[i]) > tolerance) {
            std::cerr << "Verification failed at index " << i 
                      << ": CPU = " << cpu_output[i] 
                      << ", GPU = " << gpu_output[i] 
                      << ", Diff = " << std::abs(cpu_output[i] - gpu_output[i]) << std::endl;
            return false;
        }
    }
    return true;
}
void print_array(const float* array, int size, const std::string& name){
    std::cout << name << ": ";
    for (int i = 0; i < size/10; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

// Create a Gaussian filter kernel
void create_gaussian_kernel(float* kernel, int size, float sigma) {
    float sum = 0.0f;
    int half_size = size / 2;
    
    for (int i = 0; i < size; i++) {
        int x = i - half_size;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize kernel
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

int main() {
    Profiler::print_device_properties();
    
    // Test with different data sizes
    std::vector<int> sizes = { 1<<8, 1<<10, 1<<12, 1<<14, 1<<16};
    
    for (int size : sizes) {
        std::cout << "\n=== Testing with data size: " << size << " ===" << std::endl;
        
        // Create input data (sine wave with noise)
        float* input_data = new float[size];
        for (int i = 0; i < size; i++) {
            float x = static_cast<float>(i) / size;
            input_data[i] = std::sin(2 * M_PI * 5 * x) + 0.5f * std::sin(2 * M_PI * 10 * x);
            // Add some noise
            input_data[i] += (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        }
        
        // Create Gaussian kernel
        int kernel_size = 51;
        float* kernel = new float[kernel_size];
        create_gaussian_kernel(kernel, kernel_size, 5.0f);
        
        // Allocate memory for outputs
        float* cpu_output = new float[size];
        float* gpu_output = new float[size];
        Profiler::PerformanceComparison perf("Convolution 1D (size=" + std::to_string(size) + ", kernel_size=" + std::to_string(kernel_size) + ")");
        Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
        mem_tracker.reset();
        mem_tracker.record_cpu_allocation(size * sizeof(float) + size * sizeof(float) + kernel_size * sizeof(float));
        Profiler::CPUTimer cpu_timer;
        cpu_timer.start();
        // Run CPU implementation with padding
        conv1d_cpu_padded(input_data, cpu_output, kernel, size, kernel_size);
        cpu_timer.stop();
    
        perf.set_cpu_time(cpu_timer.elapsed_milliseconds());
        // Run GPU implementation
        Profiler::KernelTimeTracker::reset();
        conv1d_gpu(input_data, gpu_output, kernel, size, kernel_size);
        perf.set_gpu_time(Profiler::KernelTimeTracker::last_total_time);
        if(size <= 1000){
            print_array(input_data, size, "Input");
            print_array(cpu_output, size, "CPU Output");
            print_array(gpu_output, size, "GPU Output");
        }
        
        // Verify results
        bool verification = verify_results(cpu_output, gpu_output, size);
        std::cout << "Verification: " << (verification ? "PASSED" : "FAILED") << std::endl;
        
        // Print performance comparison
        double gpu_time = Profiler::KernelTimeTracker::last_total_time;
        std::cout << "CPU time: " << cpu_timer.elapsed_milliseconds() << " ms" << std::endl;
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Speedup: " << cpu_timer.elapsed_milliseconds() / gpu_time << "x" << std::endl;
        
        // Clean up
        delete[] kernel;
        delete[] cpu_output;
        delete[] gpu_output;
        delete[] input_data;
    }
    
    return 0;
}