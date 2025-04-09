#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <chrono>
#include "../common/profiler.h"
#include "conv1d.h"


// Function to save data to a CSV file for visualization
bool save_csv_data(const std::string& filename, const float* data, int size) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    for (int i = 0; i < size; i++) {
        file << data[i] << std::endl;
    }
    
    file.close();
    return true;
}

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

// Create a first derivative (edge detection) kernel
void create_edge_detection_kernel(float* kernel, int size) {
    // Simple first derivative approximation
    for (int i = 0; i < size; i++) {
        if (i == 0) kernel[i] = -1.0f;
        else if (i == size - 1) kernel[i] = 1.0f;
        else kernel[i] = 0.0f;
    }
}

// Create a moving average (low-pass) kernel
void create_moving_average_kernel(float* kernel, int size) {
    float value = 1.0f / size;
    for (int i = 0; i < size; i++) {
        kernel[i] = value;
    }
}

// Generate a sample ECG signal
void generate_ecg_signal(float* signal, int size, float sample_rate) {
    // Parameters for the PQRST complex
    float p_width = 0.08f;  // seconds
    float p_height = 0.25f;
    float q_width = 0.03f;
    float q_height = -0.2f;
    float r_width = 0.03f;
    float r_height = 1.0f;
    float s_width = 0.03f;
    float s_height = -0.3f;
    float t_width = 0.16f;
    float t_height = 0.35f;
    
    float heart_rate = 75.0f;  // beats per minute
    float period = 60.0f / heart_rate;  // seconds per beat
    
    // Initialize with baseline
    for (int i = 0; i < size; i++) {
        signal[i] = 0.0f;
    }
    
    // Generate multiple heartbeats
    for (float t = 0.0f; t < size / sample_rate; t += period) {
        int start_idx = static_cast<int>(t * sample_rate);
        if (start_idx >= size) break;
        
        // P wave (Gaussian)
        float p_center = start_idx + 0.2f * sample_rate;
        for (int i = 0; i < size; i++) {
            float dt = (i - p_center) / sample_rate;
            if (std::abs(dt) < p_width) {
                signal[i] += p_height * std::exp(-dt * dt / (2 * (p_width/3) * (p_width/3)));
            }
        }
        
        // QRS complex
        float qrs_center = start_idx + 0.35f * sample_rate;
        
        // Q wave
        float q_center = qrs_center - (r_width + q_width) * sample_rate / 2;
        for (int i = 0; i < size; i++) {
            float dt = (i - q_center) / sample_rate;
            if (std::abs(dt) < q_width) {
                signal[i] += q_height * std::exp(-dt * dt / (2 * (q_width/3) * (q_width/3)));
            }
        }
        
        // R wave
        for (int i = 0; i < size; i++) {
            float dt = (i - qrs_center) / sample_rate;
            if (std::abs(dt) < r_width) {
                signal[i] += r_height * std::exp(-dt * dt / (2 * (r_width/3) * (r_width/3)));
            }
        }
        
        // S wave
        float s_center = qrs_center + (r_width + s_width) * sample_rate / 2;
        for (int i = 0; i < size; i++) {
            float dt = (i - s_center) / sample_rate;
            if (std::abs(dt) < s_width) {
                signal[i] += s_height * std::exp(-dt * dt / (2 * (s_width/3) * (s_width/3)));
            }
        }
        
        // T wave
        float t_center = start_idx + 0.5f * sample_rate;
        for (int i = 0; i < size; i++) {
            float dt = (i - t_center) / sample_rate;
            if (std::abs(dt) < t_width) {
                signal[i] += t_height * std::exp(-dt * dt / (2 * (t_width/3) * (t_width/3)));
            }
        }
    }
    
    // Add some noise
    for (int i = 0; i < size; i++) {
        // Simple random noise
        float noise = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        signal[i] += noise;
    }
}

int main() {
    Profiler::print_device_properties();
    
    std::cout << "\n=== ECG Signal Processing Example ===" << std::endl;
    
    // Parameters
    float sample_rate = 500.0f;  // Hz
    float duration = 10.0f;      // seconds
    int input_size = static_cast<int>(sample_rate * duration);
    
    // Allocate memory for input signal
    float* input_signal = new float[input_size];
    
    // Generate synthetic ECG signal
    generate_ecg_signal(input_signal, input_size, sample_rate);
    
    // Save original signal
    save_csv_data("ecg_original.csv", input_signal, input_size);
    
    // Define different filters to apply
    struct FilterTest {
        std::string name;
        int kernel_size;
        void (*create_kernel)(float*, int);
        float param;
    };
    
    std::vector<FilterTest> filters = {
        {"gaussian_smoothing", 51, [](float* k, int s) { create_gaussian_kernel(k, s, 5.0f); }, 0},
        {"edge_detection", 3, create_edge_detection_kernel, 0},
        {"moving_average", 21, create_moving_average_kernel, 0}
    };
    
    // Process with each filter
    for (const auto& filter : filters) {
        std::cout << "\nApplying " << filter.name << " filter..." << std::endl;
        
        // Create kernel
        float* kernel = new float[filter.kernel_size];
        filter.create_kernel(kernel, filter.kernel_size);
        
        // Save kernel for visualization
        save_csv_data("kernel_" + filter.name + ".csv", kernel, filter.kernel_size);
        
        // With proper padding, output size equals input size
        int output_size = input_size;
        
        // Allocate memory for outputs
        float* cpu_output = new float[output_size];
        float* gpu_output = new float[output_size];
        
        // Run CPU implementation with padding
        auto cpu_start = std::chrono::high_resolution_clock::now();
        conv1d_cpu_padded(input_signal, cpu_output, kernel, input_size, filter.kernel_size);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
        
        // Save CPU output
        save_csv_data("output_" + filter.name + "_cpu.csv", cpu_output, output_size);
        
        // Run GPU implementation
        conv1d_gpu(input_signal, gpu_output, kernel, input_size, filter.kernel_size);
        
        // Save GPU output
        save_csv_data("output_" + filter.name + "_gpu.csv", gpu_output, output_size);
        
        // Verify results
        bool verification = verify_results(cpu_output, gpu_output, output_size);
        std::cout << "Verification: " << (verification ? "PASSED" : "FAILED") << std::endl;
        
        // Print performance comparison
        double gpu_time = Profiler::KernelTimeTracker::last_total_time;
        std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Speedup: " << cpu_time.count() / gpu_time << "x" << std::endl;
        
        // Clean up
        delete[] kernel;
        delete[] cpu_output;
        delete[] gpu_output;
    }
    
    std::cout << "\nProcessing complete! CSV files generated for visualization." << std::endl;
    std::cout << "You can plot these files using Python, MATLAB, or any other plotting tool." << std::endl;
    
    // Clean up
    delete[] input_signal;
    
    return 0;
} 