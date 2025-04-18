    #include "relu.h"
    #include <iostream>
    #include <vector>
    #include "../common/profiler.h"
    #include <cstring>

    bool verify_result(const float* input, const float* output, int n, int m) {
        for (int i = 0; i < n * m; i++) {
            if (output[i] != std::max(0.0f, input[i])) {
                return false;
            }
        }
        return true;
    }

    void print_vectors(const float* input, const float* output, int n, int m) {
        std::cout << "Input: ";
        for (int i = 0; i < std::min(n * m, 10); i++) {
            std::cout << input[i] << " ";
        }
        std::cout << "\nOutput: ";  
    }   
    

    void run_test_case(int n, int m) {
        // Check if allocation size is too large
        size_t total_bytes = static_cast<size_t>(n) * m * sizeof(float);
        if (total_bytes > 2ULL * 1024 * 1024 * 1024) { // 2GB limit
            std::cout << "Skipping test case N=" << n << ", M=" << m 
                      << " - Memory requirement too large: " 
                      << (total_bytes / (1024 * 1024)) << " MB" << std::endl;
            return;
        }
        
        // Proceed with the test
        std::cout << "\n===== ReLU Test: N = " << n << ", M = " << m << " =====" << std::endl;
        Profiler::MultiApproachComparison perf("ReLU (N=" + std::to_string(n) + ", M=" + std::to_string(m) + ")");

        // Allocate 2D array using flat arrays for compatibility with existing functions
        float *input = new float[n * m];
        float *output_cpu = new float[n * m];
        float *output_gpu = new float[n * m];
        float *output_cpu_avx = new float[n * m];

        // Fill with random values between -50 and +50
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                input[i * m + j] = (static_cast<float>(rand()) / RAND_MAX * 100.0f) - 50.0f;
            }
        }
        Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
        mem_tracker.reset(); // Reset counters for this test case
        mem_tracker.record_cpu_allocation(n * m * sizeof(float) * 3); // input, output_cpu, output_gpu

        Profiler::CPUTimer cpu_timer;
        cpu_timer.start();
        relu_cpu(input, output_cpu, n, m);
        cpu_timer.stop();
        double cpu_time = cpu_timer.elapsed_milliseconds();
        perf.set_baseline_time(cpu_time);
        perf.record_approach_time("CPU Linear", cpu_time);

        Profiler::CPUTimer avx_timer;
        avx_timer.start();
        relu_cpu_avx(input, output_cpu_avx, n, m);
        avx_timer.stop();
        perf.record_approach_time("CPU AVX", avx_timer.elapsed_milliseconds()); 

        Profiler::KernelTimeTracker::reset();
        relu_gpu(input, output_gpu, n, m);
        float kernel_time = Profiler::KernelTimeTracker::get_total_kernel_time("relu") + 
                            Profiler::KernelTimeTracker::get_total_kernel_time("relu_vector4");
        perf.record_approach_time("GPU Kernel", kernel_time);

        // Print vectors for verification
        print_vectors(input, output_cpu, n, m);
        std::cout << "GPU Output: ";
        for(int i = 0; i < std::min(n * m, 10); i++){
            std::cout << output_gpu[i] << " ";
        }
        std::cout << "\nCPU AVX Output: ";
        for(int i = 0; i < std::min(n * m, 10); i++){
            std::cout << output_cpu_avx[i] << " ";
        }
        std::cout << std::endl;

        // Verify results
        perf.set_approach_verified("CPU Linear", verify_result(input, output_cpu, n, m));
        perf.set_approach_verified("GPU Kernel", verify_result(input, output_gpu, n, m));
        perf.set_approach_verified("CPU AVX", verify_result(input, output_cpu_avx, n, m));
        perf.print_summary();
        std::cout << "GPU Kernel Time Breakdown: " << std::endl;
        for (const auto& kernel_name : {"relu", "relu_vector4"}) {
            std::cout << "Kernel: " << kernel_name << " Time: " << Profiler::KernelTimeTracker::get_total_kernel_time(kernel_name) << " ms" << std::endl;
        }
        mem_tracker.print_summary();

        //   Clean up
        delete[] input;
        delete[] output_cpu;
        delete[] output_gpu;
        delete[] output_cpu_avx;
    }

    int main() {
        Profiler::print_device_properties();

        std::vector<std::pair<int, int>> test_sizes = {
            {1 << 10, 1 << 10},  // 1M elements
            {1 << 12, 1 << 12},  // 16M elements
            {1 << 14, 1 << 14},  // 256M elements 
            {1 << 16, 1 << 10},  // 64M elements (rectangular)
            {1 << 18, 1 << 8}    // 64M elements (rectangular)
        };
        
        for (auto size : test_sizes) {
            run_test_case(size.first, size.second);
        }

        return 0;
    }