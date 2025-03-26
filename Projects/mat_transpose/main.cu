#include "matTranspose.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "../common/profiler.h"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <tuple>
#include <algorithm>

void generate_random_matrix(float* matrix, int rows, int cols) {
    std::mt19937 gen(std::random_device{}());
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
        std::cout << std::setw(8) << std::fixed << std::setprecision(2) << matrix[i * cols];
        for (int j = 1; j < print_cols; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << matrix[i * cols + j];
        }
        std::cout << std::endl;
    }
}

bool verify_result(float* c_cpu, float* c_gpu, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        if (std::abs(c_cpu[i] - c_gpu[i]) > 1e-2) {
            std::cout << "Verification failed at index " << i << ": CPU=" << c_cpu[i] << ", GPU=" << c_gpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

void run_test_case(int rows, int cols) {
    std::cout << "Running test case with size: " << rows << "x" << cols << std::endl;
    float* input = new float[rows * cols];
    float* output = new float[cols * rows];
    float* c_cpu = new float[cols * rows];
    float* c_gpu = new float[cols * rows];

    generate_random_matrix(input, rows, cols);
    std::fill(c_cpu, c_cpu + cols * rows, 0);
    std::fill(c_gpu, c_gpu + cols * rows, 0);
    Profiler::PerformanceComparison perf("Matrix Transpose (M=" + std::to_string(rows) + ", N=" + std::to_string(cols) + ")");
    Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    mem_tracker.reset();
    mem_tracker.record_cpu_allocation(rows * cols * sizeof(float) + cols * rows * sizeof(float) * 2);
    Profiler::CPUTimer cpu_timer;
    cpu_timer.start();
    matrix_transpose_cpu(input, c_cpu, rows, cols);
    cpu_timer.stop();
    perf.set_cpu_time(cpu_timer.elapsed_milliseconds());

    Profiler::KernelTimeTracker::reset();
    matrix_transpose_gpu(input, c_gpu, rows, cols);
    perf.set_gpu_time(Profiler::KernelTimeTracker::last_total_time);

    if(rows <= 32 && cols <= 32){
        print_matrix(input, rows, cols, "Input");
        print_matrix(c_cpu, cols, rows, "CPU Result");
        print_matrix(c_gpu, cols, rows, "GPU Result");
    }
    bool verified = verify_result(c_cpu, c_gpu, cols, rows);
    perf.set_verified(verified);
    if(verified){
        std::cout << "Verification passed" << std::endl;
    }
    perf.print_summary();
    mem_tracker.print_summary();
    delete[] input;
    delete[] output;
    delete[] c_cpu;
    delete[] c_gpu;
}

int main() {
    Profiler::print_device_properties();
    std::vector<std::tuple<int, int>> test_cases = {
        {1, 1},
        {1023, 1023},
        {1 << 10, 1 << 10},
        {1 << 12, 1 << 12},
        {1 << 14, 1 << 14},
        // {1 << 16, 1 << 16},
        // {1 << 18, 1 << 18},
    };

    for (const auto& test : test_cases) {
        int rows = std::get<0>(test);
        int cols = std::get<1>(test);
        run_test_case(rows, cols);
    }
    run_test_case(1, 1);
    run_test_case(1000, 1);
    std::cout << "\nALL TESTS COMPLETED SUCCESSFULLY\n";
    return 0;
}