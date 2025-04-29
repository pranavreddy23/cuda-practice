#include "conv2D.h"
#include <iostream>
#include <vector>
#include "../common/profiler.h"
#include <cstring>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <tuple>
#include <filesystem>
#include <string>
#include <algorithm> // For std::fill
// Include stb_image for image loading/saving
#define STB_IMAGE_IMPLEMENTATION
#include "../common/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../common/stb_image_write.h"
namespace fs = std::filesystem;

void gaussian_kernel_init(unsigned char* kernel, int kernel_size) {
    float sigma = kernel_size / 2.0f;
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            float x = i - kernel_size / 2;
            float y = j - kernel_size / 2;
            float gaussian = exp(-(x*x + y*y) / (2*sigma*sigma)) / (2*M_PI*sigma*sigma);
            kernel[i*kernel_size + j] = gaussian;
            sum += gaussian;
        }
    }
    for (int i = 0; i < kernel_size; i++) { 
        for (int j = 0; j < kernel_size; j++) {
            kernel[i*kernel_size + j] /= sum;
        }
    }
}

void process_image(const std::string& image_path , int kernel_size) {
    std::cout << "Processing image: " << image_path << " with kernel size " << kernel_size << "x" << kernel_size << std::endl;

    fs::path path(image_path);
    std::string filename = path.stem().string();
    std::string extension = path.extension().string();
    std::string cpu_output_path = "output/" + filename + "_cpu" + extension;
    std::string gpu_output_path = "output/" + filename + "_gpu" + extension;
    std::string avx_output_path = "output/" + filename + "_avx" + extension; // Added AVX path


    int width, height, channels;
    // Load image, force 4 channels (RGBA) for consistency maybe? Or handle channels properly later.
    // Let's assume grayscale for simplicity for now based on conv code. Force 1 channel.
    unsigned char* image = stbi_load(image_path.c_str(), &width, &height, &channels, 1);

    if (!image) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return;
    }
    std::cout << "Image loaded: " << width << "x" << height << " (Loaded as 1 channel)" << std::endl;
    channels = 1; // Force channels to 1 since conv code assumes it

    // --- Padding Setup ---
    if (kernel_size % 2 == 0) {
        std::cerr << "Warning: Kernel size should be odd for symmetric padding. Using " << kernel_size << std::endl;
    }
    int pad_rows = (kernel_size - 1) / 2;
    int pad_cols = (kernel_size - 1) / 2;
    int padded_width = width + 2 * pad_cols;
    int padded_height = height + 2 * pad_rows;
    size_t original_image_size = static_cast<size_t>(width) * height * channels;
    size_t padded_image_size = static_cast<size_t>(padded_width) * padded_height * channels;

    // Allocate buffers
    unsigned char* padded_input_cpu = new unsigned char[padded_image_size];
    unsigned char* padded_input_gpu = new unsigned char[padded_image_size]; // Need separate one for GPU call prep
    unsigned char* cpu_output = new unsigned char[original_image_size]; // Output is original size
    unsigned char* gpu_output = new unsigned char[original_image_size]; // Output is original size
    unsigned char* cpu_output_avx = new unsigned char[original_image_size]; // Output is original size
    unsigned char* kernel = new unsigned char[kernel_size * kernel_size];

    // Initialize kernel (assuming gaussian_kernel_init works for unsigned char)
    // gaussian_kernel_init(kernel, kernel_size); // Needs modification for unsigned char scaling

    // --- Create Padded Input ---
    // Fill padding with zeros (or replicate border pixels if preferred)
    std::fill(padded_input_cpu, padded_input_cpu + padded_image_size, 0);
    // Copy original image into the center of the padded buffer
    for (int i = 0; i < height; ++i) {
        std::memcpy(
            padded_input_cpu + (i + pad_rows) * padded_width * channels + pad_cols * channels, // Destination start in padded
            image + i * width * channels,              // Source start in original
            width * channels * sizeof(unsigned char)   // Bytes per row
        );
    }
    // Copy the same padded data for GPU prep
     std::memcpy(padded_input_gpu, padded_input_cpu, padded_image_size);


    // --- Profiling Setup ---
    Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    mem_tracker.reset();
    mem_tracker.record_cpu_allocation(padded_image_size * 2 + original_image_size * 3 + kernel_size * kernel_size); // padded x2, output x3, kernel
    Profiler::MultiApproachComparison perf("Convolution 2D (" + filename + ")");

    // --- Run CPU Convolution ---
    Profiler::CPUTimer cpu_timer;
    cpu_timer.start();
    // Pass padded input, original dimensions
    conv2D_cpu(padded_input_cpu, kernel, cpu_output, width, height, kernel_size, kernel_size);
    cpu_timer.stop();
    double cpu_time = cpu_timer.elapsed_milliseconds();
    perf.set_baseline_time(cpu_time);
    perf.record_approach_time("CPU Linear", cpu_time);

    // --- Run CPU AVX Convolution --- (Needs modification for padding)
    // Profiler::CPUTimer avx_timer;
    // avx_timer.start();
    // conv2D_cpu_avx(padded_input_cpu, kernel, cpu_output_avx, width, height, kernel_size, kernel_size); // Adapt this call
    // avx_timer.stop();
    // double cpu_time_avx = avx_timer.elapsed_milliseconds();
    // perf.record_approach_time("CPU AVX", cpu_time_avx);
    std::cout << "Skipping AVX for now (needs padding update)." << std::endl;


    // --- Run GPU Convolution ---
    Profiler::KernelTimeTracker::reset();
    // Pass the host padded input (conv2D_gpu handles device transfer)
    conv2D_gpu(padded_input_gpu, kernel, gpu_output, width, height, kernel_size, kernel_size);
    float kernel_time = Profiler::KernelTimeTracker::get_total_kernel_time("conv2D_padded"); // Use the new name
                       // + Profiler::KernelTimeTracker::get_total_kernel_time("conv2D_vector4"); // Remove if not using vector4
    perf.record_approach_time("GPU Kernel", kernel_time);


    // --- Save Results ---
    fs::create_directory("output");
    // Save outputs (which are original size)
    stbi_write_png(cpu_output_path.c_str(), width, height, channels, cpu_output, width * channels);
    stbi_write_png(gpu_output_path.c_str(), width, height, channels, gpu_output, width * channels);
    // stbi_write_png(avx_output_path.c_str(), width, height, channels, cpu_output_avx, width * channels); // If AVX is done


    // --- Print Summary ---
    perf.print_summary();
    Profiler::KernelTimeTracker::print_kernel_times(); // Will show 'conv2D_padded'
    mem_tracker.print_summary();


    // --- Clean up ---
    stbi_image_free(image);
    delete[] padded_input_cpu;
    delete[] padded_input_gpu;
    delete[] cpu_output;
    delete[] gpu_output;
    delete[] cpu_output_avx;
    delete[] kernel;
}

int main(int argc, char** argv) {
    Profiler::print_device_properties();
    fs::create_directory("output");
    int kernel_size = 5; // Example kernel size

    if (argc > 1) {
        // Allow setting kernel size from command line? e.g., ./conv2d 7 image.png
         if (argc > 2 && (std::string(argv[1]) == "-k" || std::string(argv[1]) == "--kernel")) {
             kernel_size = std::stoi(argv[2]);
             if (kernel_size % 2 == 0) {
                 std::cerr << "Error: Kernel size must be odd." << std::endl;
                 return 1;
             }
             std::cout << "Using kernel size: " << kernel_size << std::endl;
             for (int i = 3; i < argc; i++) { // Process images after kernel flag
                process_image(argv[i], kernel_size);
            }
         } else { // No kernel flag, assume all args are images
            for (int i = 1; i < argc; i++) {
                process_image(argv[i], kernel_size);
            }
         }

    } else {
        std::string image_dir = "images";
        if (!fs::exists(image_dir)) {
            std::cerr << "Images directory not found. Please create an 'images' directory and add some images." << std::endl;
            return 1;
        }
        for (const auto& entry : fs::directory_iterator(image_dir)) {
             if (entry.is_regular_file()) {
                 std::string extension = entry.path().extension().string();
                 std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower); // Lowercase extension
                 if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
                     extension == ".bmp" || extension == ".tga") {
                     process_image(entry.path().string(), kernel_size);
                 }
             }
         }
    }

    std::cout << "All images processed successfully!" << std::endl;
    return 0;
}
   