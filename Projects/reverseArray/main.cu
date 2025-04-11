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
    
    // Allocate memory for input and output arrays
    float* input = new float[size];
    float* cpu_output = new float[size];
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

    // Create performance comparison object
    Profiler::PerformanceComparison perf("Reverse Array (N=" + std::to_string(size) + ")");

    // Track memory usage
    Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    mem_tracker.reset(); // Reset counters for this test case
    mem_tracker.record_cpu_allocation(size * sizeof(float) * 4); // input, cpu_output, gpu_output, input_copy
    
    // Run and time CPU implementation
    Profiler::CPUTimer cpu_timer;
    cpu_timer.start();
    
    // Copy input to cpu_output first
    std::memcpy(cpu_output, input, size * sizeof(float));
    reverseArray_cpu(cpu_output, size);
    
    cpu_timer.stop();
    perf.set_cpu_time(cpu_timer.elapsed_milliseconds());

    // Run and time GPU implementation
    Profiler::KernelTimeTracker::reset();
    
    // Copy input to gpu_output first
    std::memcpy(gpu_output, input, size * sizeof(float));
    reverseArray_gpu(gpu_output, size);
    
    perf.set_gpu_time(Profiler::KernelTimeTracker::last_total_time);

    // Print vectors for verification
    print_vectors(input, cpu_output, size);
    std::cout << "GPU Output: ";
    for(int i = 0; i < std::min(size, 10); i++){
        std::cout << gpu_output[i] << " ";
    }
    std::cout << std::endl;

    // Verify results for both CPU and GPU
    bool verified_cpu = verify_result(input_copy, cpu_output, size);
    bool verified_gpu = verify_result(input_copy, gpu_output, size);
    perf.set_verified(verified_cpu && verified_gpu);

    // Print performance summary
    perf.print_summary();
    mem_tracker.print_summary();

    // Clean up memory - make sure each pointer is freed exactly once
    delete[] input;
    delete[] cpu_output;    
    delete[] gpu_output;
    delete[] input_copy;
}

int main(){
    Profiler::print_device_properties();
    std::vector<int> sizes = {1<<8, 1<<10, 1<<12, 1<<14, 1<<16};
    for(int size : sizes){
        run_test_case(size);
    }
    std::cout << "\nALL TESTS COMPLETED SUCCESSFULLY\n";
    return 0;
}