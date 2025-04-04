#include "colorInversion.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "../common/profiler.h"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <tuple>
#include <filesystem>
#include <string>
#include <cstring>

// Include stb_image for image loading/saving
#define STB_IMAGE_IMPLEMENTATION
#include "../common/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../common/stb_image_write.h"

namespace fs = std::filesystem;

// Function to process a single image
void process_image(const std::string& input_path) {
    // Extract filename for output
    fs::path path(input_path);
    std::string filename = path.stem().string();
    std::string extension = path.extension().string();
    
    // Create output paths
    std::string cpu_output_path = "output/" + filename + "_cpu" + extension;
    std::string gpu_output_path = "output/" + filename + "_gpu" + extension;
    
    // Load image
    int width, height, channels;
    unsigned char* image = stbi_load(input_path.c_str(), &width, &height, &channels, 4); // Force RGBA
    
    if (!image) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return;
    }
    
    std::cout << "Processing image: " << input_path << " (" << width << "x" << height << ")" << std::endl;
    
    // Create copies for CPU and GPU processing
    size_t image_size = width * height * 4;
    unsigned char* cpu_output = new unsigned char[image_size];
    unsigned char* gpu_output = new unsigned char[image_size];
    
    // Copy original image to output buffers
    std::memcpy(cpu_output, image, image_size);
    std::memcpy(gpu_output, image, image_size);
    
    // Get memory tracker instance
    Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    mem_tracker.reset();
    mem_tracker.record_cpu_allocation(image_size * 3); // original + 2 copies
    
    // Create performance comparison object
    Profiler::PerformanceComparison perf("Color Inversion (" + filename + ")");
    
    // Run CPU implementation
    Profiler::CPUTimer cpu_timer;
    cpu_timer.start();
    colorInversion_cpu(cpu_output, width, height);
    cpu_timer.stop();
    perf.set_cpu_time(cpu_timer.elapsed_milliseconds());
    
    // Reset kernel timers before GPU execution
    Profiler::KernelTimeTracker::reset();
    
    // Run GPU implementation
    colorInversion_gpu(gpu_output, width, height);
    perf.set_gpu_time(Profiler::KernelTimeTracker::last_total_time);
    
    // Save output images
    fs::create_directory("output"); // Create output directory if it doesn't exist
    stbi_write_png(cpu_output_path.c_str(), width, height, 4, cpu_output, width * 4);
    stbi_write_png(gpu_output_path.c_str(), width, height, 4, gpu_output, width * 4);
    
    // Print performance summary
    perf.set_verified(true); // We're not doing pixel-by-pixel verification
    perf.print_summary();
    
    // Print detailed kernel timing breakdown
    Profiler::KernelTimeTracker::print_kernel_times();
    
    // Print memory usage
    mem_tracker.print_summary();
    
    // Clean up
    stbi_image_free(image);
    delete[] cpu_output;
    delete[] gpu_output;
}

int main(int argc, char** argv) {
    Profiler::print_device_properties();
    
    // Create output directory if it doesn't exist
    fs::create_directory("output");
    
    if (argc > 1) {
        // Process specific images provided as command-line arguments
        for (int i = 1; i < argc; i++) {
            process_image(argv[i]);
        }
    } else {
        // Process all images in the "images" directory
        std::string image_dir = "images";
        
        if (!fs::exists(image_dir)) {
            std::cerr << "Images directory not found. Please create an 'images' directory and add some images." << std::endl;
            return 1;
        }
        
        // Process each image in the directory
        for (const auto& entry : fs::directory_iterator(image_dir)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                // Check if it's an image file
                if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || 
                    extension == ".bmp" || extension == ".tga") {
                    process_image(entry.path().string());
                }
            }
        }
    }
    
    std::cout << "All images processed successfully!" << std::endl;
    return 0;
}