#include "softmax.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "../common/profiler.h"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <tuple>
#include <random>

void generate_random_vector(float* vec, int N) {
    static std::mt19937 engine(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    
    for(int i = 0; i < N; i++) {
        vec[i] = dist(engine);
    }
}

void print_vector(float* vec, int N, const std::string& name){
    std::cout << name << " (" << N << "):" << std::endl;
    std::cout <<name<<std::endl;
    for(int i = 0; i < N; i++){
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

bool verify_result(float* vec1, float* vec2, int N){
    for(int i = 0; i < N; i++){
        if(std::abs(vec1[i] - vec2[i]) > 1e-2){
            std::cout << "Verification failed at index " << i << ": CPU=" << vec1[i] << ", GPU=" << vec2[i] << std::endl;
            std::cout<<"CPU Result: "<<vec1[i]<<std::endl;
            std::cout<<"GPU Result: "<<vec2[i]<<std::endl;
            return false;
        }
    }
    return true;
}
void run_test_case(int N) {
    std::cout << "Running test case with size: " << N << std::endl;
    float* input = new float[N];
    float* output_cpu = new float[N];
    float* output_gpu = new float[N];
    generate_random_vector(input, N);
    Profiler::PerformanceComparison perf("Softmax (N=" + std::to_string(N) + ")");
    Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    mem_tracker.reset();
    mem_tracker.record_cpu_allocation(sizeof(float) * N);
    Profiler::CPUTimer cpu_timer;
    cpu_timer.start();
    softmax_cpu(input, output_cpu, N);
    cpu_timer.stop();
    perf.set_cpu_time(cpu_timer.elapsed_milliseconds());
    Profiler::KernelTimeTracker::reset();
    softmax_gpu(input, output_gpu, N);
    perf.set_gpu_time(Profiler::KernelTimeTracker::last_total_time);
    if(N <= 32){
        print_vector(input, N, "Input");
        print_vector(output_cpu, N, "Output CPU");
        print_vector(output_gpu, N, "Output GPU");
    }
    bool verified = verify_result(output_cpu, output_gpu, N);
    perf.set_verified(verified);
    if(verified){
        std::cout << "Verification passed" << std::endl;
    }
    perf.print_summary();
    
    Profiler::KernelTimeTracker::print_kernel_times();
    mem_tracker.print_summary();
    delete[] input;
    delete[] output_cpu;
    delete[] output_gpu;
}

int main() {
    Profiler::print_device_properties();
    std::vector<int> test_sizes = {1<<2, 1 << 3, 1<<4, 1 << 10, 1 << 12, 1 << 14, 1 << 16};
    for (int size : test_sizes) {
        run_test_case(size);

    }
    run_test_case(1);
    run_test_case(1 << 1023);
    std::cout<< "ALL TEST CASES DONE SUCCESSFULLY" << std::endl;    
    return 0;
}