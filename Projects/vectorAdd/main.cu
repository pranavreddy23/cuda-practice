#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "../common/profiler.h"
#include "vectorAdd.h"

// Check vector add result
bool verify_result(const std::vector<int> &a, const std::vector<int> &b,
                   const std::vector<int> &c, int N) {
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) {
            return false;
        }
    }
    return true;
}

// Print vectors for verification
void print_vectors(const std::vector<int> &a, const std::vector<int> &b, 
                   const std::vector<int> &c_cpu, const std::vector<int> &c_gpu, int N) {
    std::cout << "Vector A: ";
    for (int i = 0; i < std::min(N, 10); i++) { // Print first 10 elements
        std::cout << a[i] << " ";
    }
    std::cout << "\nVector B: ";
    for (int i = 0; i < std::min(N, 10); i++) { // Print first 10 elements
        std::cout << b[i] << " ";
    }
    std::cout << "\nResult Vector C (CPU): ";
    for (int i = 0; i < std::min(N, 10); i++) { // Print first 10 elements
        std::cout << c_cpu[i] << " ";
    }
    std::cout << "\nResult Vector C (GPU): ";
    for (int i = 0; i < std::min(N, 10); i++) { // Print first 10 elements
        std::cout << c_gpu[i] << " ";
    }
    std::cout << std::endl;
}

// Run a test case with both CPU and GPU implementations
void run_test_case(int N) {
    std::cout << "\n===== Vector Addition Test: N = " << N << " =====" << std::endl;
    
    // Create vectors for test data
    std::vector<int> a(N);
    std::vector<int> b(N);
    std::vector<int> c_cpu(N);
    std::vector<int> c_gpu(N);
    
    // Initialize with random data
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }
    
    // Create performance comparison object
    Profiler::PerformanceComparison perf("Vector Addition (N=" + std::to_string(N) + ")");
    
    // Track memory usage
    Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    mem_tracker.reset(); // Reset counters for this test case
    mem_tracker.record_cpu_allocation(N * sizeof(int) * 4); // a, b, c_cpu, c_gpu
    
    // Run and time CPU implementation
    Profiler::CPUTimer cpu_timer;
    cpu_timer.start();
    vectorAdd_cpu(a, b, c_cpu, N);
    cpu_timer.stop();
    perf.set_cpu_time(cpu_timer.elapsed_milliseconds());
    
    // Run and time GPU implementation
    Profiler::GPUTimer gpu_timer;
    gpu_timer.start();
    vectorAdd_gpu(a, b, c_gpu, N);
    gpu_timer.stop();
    perf.set_gpu_time(gpu_timer.elapsed_milliseconds());
    
    // Print vectors for verification (after they've been populated)
    print_vectors(a, b, c_cpu, c_gpu, N);
    
    // Verify results for both CPU and GPU
    bool verified_cpu = verify_result(a, b, c_cpu, N);
    bool verified_gpu = verify_result(a, b, c_gpu, N);
    
    if (!verified_cpu) {
        std::cout << "CPU verification FAILED!" << std::endl;
    }
    
    if (!verified_gpu) {
        std::cout << "GPU verification FAILED!" << std::endl;
    }
    
    perf.set_verified(verified_cpu && verified_gpu);
    
    // Print performance summary
    perf.print_summary();
    mem_tracker.print_summary();
}

int main() {
    // Print CUDA device information
    Profiler::print_device_properties();
    
    // Run tests with different sizes
    std::vector<int> test_sizes = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18}; // 1K to 256K elements
    for (int size : test_sizes) {
        run_test_case(size);
    }
    
    // Edge cases
    run_test_case(1);        // Single element
    run_test_case(1023);     // Non-power of 2
    
    std::cout << "\nALL TESTS COMPLETED SUCCESSFULLY\n";
    return 0;
}
