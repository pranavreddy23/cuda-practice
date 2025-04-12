#include "reverseArray.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "../common/profiler.h"
#include <cassert>  
#include <cstring>

bool verify_result(const float* input, const float* output, int size){
    for(int i = 0; i < size; i++){
        if(input[i] != output[size - i - 1]){
            return false;
        }
    }
    return true;
}

void print_vectors(const float* input, const float* output, int size){
    std::cout << "\nInput: ";
    for(int i = 0; i < std::min(size, 10); i++){
        std::cout << input[i] << " ";
    }
    std::cout << "\nOutput: ";
    for(int i = 0; i < std::min(size, 10); i++){
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
}

void run_test_case(int size){
    std::cout << "\n===== Reverse Array Test: N = " << size << " =====" << std::endl;
    
    // Create multi-approach comparison object
    Profiler::MultiApproachComparison perf("Reverse Array (N=" + std::to_string(size) + ")");
    
    // Allocate memory for input and output arrays
    float* input = new float[size];
    float* cpu_output = new float[size];
    float* avx_output = new float[size];
    float* gpu_output = new float[size];
    
    // Create a copy of the input for verification
    float* input_copy = new float[size];

    // Initialize input array with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for(int i = 0; i < size; i++){
        input[i] = dis(gen);
    }
    
    // Make a copy of the input for verification
    std::memcpy(input_copy, input, size * sizeof(float));

    // Track memory usage
    Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    mem_tracker.reset(); // Reset counters for this test case
    mem_tracker.record_cpu_allocation(size * sizeof(float) * 5); // input, input_copy, cpu_output, avx_output, gpu_output
    
    // Run and time CPU implementation (baseline)
    Profiler::CPUTimer cpu_timer;
    cpu_timer.start();
    std::memcpy(cpu_output, input, size * sizeof(float));
    reverseArray_cpu(cpu_output, size);
    cpu_timer.stop();
    double cpu_time = cpu_timer.elapsed_milliseconds();
    perf.set_baseline_time(cpu_time);
    perf.record_approach_time("CPU Linear", cpu_time);
    
    // Run and time AVX implementation
    Profiler::CPUTimer avx_timer;
    avx_timer.start();
    std::memcpy(avx_output, input, size * sizeof(float));
    reverseArray_avx(avx_output, size);
    avx_timer.stop();
    perf.record_approach_time("CPU AVX", avx_timer.elapsed_milliseconds());
    
    // Run and time GPU implementation
    Profiler::KernelTimeTracker::reset();
    std::memcpy(gpu_output, input, size * sizeof(float));
    reverseArray_gpu(gpu_output, size);
    
    // Only record the kernel time, not the total GPU time
    float kernel_time = Profiler::KernelTimeTracker::get_total_kernel_time("reverseArray");
    perf.record_approach_time("GPU Kernel", kernel_time);
    
    // Print vectors for verification
    print_vectors(input, cpu_output, size);
    std::cout << "AVX Output: ";
    for(int i = 0; i < std::min(size, 10); i++){
        std::cout << avx_output[i] << " ";
    }
    std::cout << "\nGPU Output: ";
    for(int i = 0; i < std::min(size, 10); i++){
        std::cout << gpu_output[i] << " ";
    }
    std::cout << std::endl;

    // Verify results
    perf.set_approach_verified("CPU Linear", verify_result(input_copy, cpu_output, size));
    perf.set_approach_verified("CPU AVX", verify_result(input_copy, avx_output, size));
    perf.set_approach_verified("GPU Kernel", verify_result(input_copy, gpu_output, size));
    
    // Print performance summary
    perf.print_summary();
    
    // Print kernel times with speedups
    std::cout << "GPU Kernel Timing Breakdown:" << std::endl;
    for (const auto& kernel_name : {"reverseArray"}) {
        float kernel_time = Profiler::KernelTimeTracker::get_total_kernel_time(kernel_name);
        double speedup = cpu_time / kernel_time;
        std::cout << "  " << kernel_name << ": " << kernel_time << " ms (Speedup: " << speedup << "x)" << std::endl;
    }
    
    mem_tracker.print_summary();

    // Clean up memory
    delete[] input;
    delete[] cpu_output;
    delete[] avx_output;
    delete[] gpu_output;
    delete[] input_copy;
}

int main(){
    Profiler::print_device_properties();
    std::vector<int> sizes = {1<<8, 1<<10, 1<<12, 1<<14, 1<<16, 1<<20};
    for(int size : sizes){
        run_test_case(size);
    }
    std::cout << "\nALL TESTS COMPLETED SUCCESSFULLY\n";
    return 0;
}