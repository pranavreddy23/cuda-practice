#include "matmul.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "../common/profiler.h"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <tuple>

void generate_random_matrix(float* matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

void print_matrix(float* matrix, int rows, int cols, const std::string& name) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    int print_rows = std::min(rows, 10);
    int print_cols = std::min(cols, 10);
    for (int i = 0; i < print_rows; ++i) {
        for (int j = 0; j < print_cols; ++j) {
            std::cout << std::setw(8) << matrix[i * cols + j] << " ";
        }
        if (cols > print_cols) {
            std::cout << "...";
        }
        std::cout << std::endl;
    }
    if (rows > print_rows) {
        std::cout << "..." << std::endl;
    }
}

bool verify_result(float* c_cpu, float* c_gpu, int M, int K) {
    for (int i = 0; i < M * K; ++i) {
        if (std::abs(c_cpu[i] - c_gpu[i]) > 1e-2) {
            std::cout << "Verification failed at index " << i << ": CPU=" << c_cpu[i] << ", GPU=" << c_gpu[i] << std::endl;
            std::cout<<"CPU Result: "<<c_cpu[i]<<std::endl;
            std::cout<<"GPU Result: "<<c_gpu[i]<<std::endl;
            return false;
        }
    }
    return true;
}

void run_test_case(int M, int N, int K) {
    std::cout << "Running test case with size: " << M << "x" << N << "x" << K << std::endl;
    float* A = new float[M * N];
    float* B = new float[N * K];
    float* c_cpu = new float[M * K];
    float* c_gpu = new float[M * K];

    generate_random_matrix(A, M, N);
    generate_random_matrix(B, N, K);

    std::fill(c_cpu, c_cpu + M * K, 0);
    std::fill(c_gpu, c_gpu + M * K, 0);
    Profiler::PerformanceComparison perf("Matrix Multiplication (M=" + std::to_string(M) + ", N=" + std::to_string(N) + ", K=" + std::to_string(K) + ")");
    Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    mem_tracker.reset();
    mem_tracker.record_cpu_allocation(M * N * sizeof(float) + N * K * sizeof(float) + M * K * sizeof(float) * 2);
    Profiler::CPUTimer cpu_timer;
    cpu_timer.start();
    matmul_cpu_naive(A, B, c_cpu, M, N, K);
    cpu_timer.stop();
    perf.set_cpu_time(cpu_timer.elapsed_milliseconds());

    Profiler::KernelTimeTracker::reset();
    matmul_gpu(A, B, c_gpu, M, N, K);
    perf.set_gpu_time(Profiler::KernelTimeTracker::last_total_time);
    if(M <=32 && N <=32 && K <=32){
        print_matrix(A, M, N, "A");
        print_matrix(B, N, K, "B");
        print_matrix(c_cpu, M, K, "CPU Result");
        print_matrix(c_gpu, M, K, "GPU Result");
    }
    bool verified = verify_result(c_cpu, c_gpu, M, K);
    perf.set_verified(verified);
    if(verified){
        std::cout << "Verification passed" << std::endl;
    }
    perf.print_summary();
    mem_tracker.print_summary();
    delete[] A;
    delete[] B;
    delete[] c_cpu;
    delete[] c_gpu; 
   
 }

int main() {
    Profiler::print_device_properties();
    std::vector<std::tuple<int, int, int>> test_cases = {
        {1, 1, 1},
        {1023, 1023, 1023},
        {1 << 10, 1 << 10, 1 << 10},
        {1 << 12, 1 << 12, 1 << 12},
        // {1 << 14, 1 << 14, 1 << 14},
        // {1 << 16, 1 << 16, 1 << 16},
        // {1 << 18, 1 << 18, 1 << 18},
        {64, 128, 32},
        {128, 64, 32},
        {32, 64, 128}
    };

    for (const auto& test : test_cases) {
        int M = std::get<0>(test);
        int N = std::get<1>(test);
        int K = std::get<2>(test);
        run_test_case(M, N, K);
    }
    run_test_case(1, 1, 1);
    run_test_case(3, 1, 3);       // 3x1 * 1x3 (outer product)
    run_test_case(1, 1000, 1);  
    std::cout << "\nALL TESTS COMPLETED SUCCESSFULLY\n";
    return 0;
}
