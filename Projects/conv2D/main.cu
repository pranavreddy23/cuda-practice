#include "conv2D.h"
#include <iostream>
#include <vector>
#include "../common/profiler.h"
#include <cstring>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <tuple>
#include <string>
#include <algorithm> // For std::fill

// --- OpenCV Includes ---
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp> // For imread/imwrite
#include <opencv2/imgproc.hpp>  // For copyMakeBorder

// --- Remove STB Includes ---
// #define STB_IMAGE_IMPLEMENTATION
// #include "../common/stb_image.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "../common/stb_image_write.h"

// --- Remove Filesystem Includes (optional, can keep if needed elsewhere) ---
// #include <filesystem>
// namespace fs = std::filesystem; // Remove or replace uses

void gaussian_kernel_init(unsigned char* kernel, int kernel_size) {
    // Use a fixed sigma like your reference code, or calculate based on kernel size
    float sigma = 1.0f; // Try this fixed value first
    float s = 2.0f * sigma * sigma;
    
    // For storing the floating-point values before conversion
    float* temp_kernel = new float[kernel_size * kernel_size];
    float sum = 0.0f;
    
    int radius = kernel_size / 2;
    
    // Generate the Gaussian kernel with proper coordinates
    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            // Calculate Euclidean distance from center
            float r = sqrt(x*x + y*y);
            
            // Apply Gaussian formula - note this matches your reference exactly
            float value = (exp(-(r*r) / s)) / (M_PI * s);
            
            // Store in the temp kernel (convert to 0-based indices)
            temp_kernel[(x + radius) * kernel_size + (y + radius)] = value;
            sum += value;
        }
    }
    
    // Normalize the kernel
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        temp_kernel[i] /= sum;
    }
    
    // Scale to 0-255 range for unsigned char
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel[i] = static_cast<unsigned char>(temp_kernel[i] * 255.0f);
    }
    
    delete[] temp_kernel;
}

void box_blur_kernel_init(unsigned char* kernel, int kernel_size) {
    // Box blur gives equal weight to all pixels
    float weight = 1.0f / (kernel_size * kernel_size);
    
    // Scale to 0-255 range
    unsigned char value = static_cast<unsigned char>(weight * 255.0f);
    
    // Fill kernel with the same value
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel[i] = value;
    }
    
    std::cout << "Box blur kernel value: " << (int)value << std::endl;
}

void identity_kernel_init(unsigned char* kernel, int kernel_size) {
    // Set all values to 0
    std::fill(kernel, kernel + kernel_size * kernel_size, 0);
    
    // Set center pixel to 255 (full weight)
    int center = kernel_size / 2;
    kernel[center * kernel_size + center] = 255;
    
    std::cout << "Identity kernel created" << std::endl;
}

void process_image(const std::string& image_path, int kernel_size) {
    std::cout << "Processing image: " << image_path << " with kernel size " << kernel_size << "x" << kernel_size << std::endl;

    // --- Use OpenCV to load the image ---
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR); // Load as BGR

    if (image.empty()) {
        std::cerr << "Failed to load image using OpenCV: " << image_path << std::endl;
        return;
    }

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels(); // Should be 3 for BGR

    if (channels != 3) {
         std::cerr << "Warning: Image loaded does not have 3 channels. Found " << channels << ". Convolution might not work as expected." << std::endl;
         // You might want to handle grayscale or convert here if necessary
         // For now, we'll proceed assuming 3 channels might work, but results could be odd.
    }

    std::cout << "Image loaded: " << width << "x" << height << " (" << channels << " channels)" << std::endl;

    // --- Basic Filename Extraction (without std::filesystem) ---
    size_t last_slash_idx = image_path.find_last_of("\\/");
    std::string base_filename = (std::string::npos == last_slash_idx) ? image_path : image_path.substr(last_slash_idx + 1);
    size_t dot_idx = base_filename.find_last_of('.');
    std::string filename_no_ext = (std::string::npos == dot_idx) ? base_filename : base_filename.substr(0, dot_idx);
    std::string extension = (std::string::npos == dot_idx) ? "" : base_filename.substr(dot_idx);
    
    // --- Create output directory (simple way, consider platform differences) ---
    // For Linux/macOS:
    system("mkdir -p output"); 
    // For Windows:
    // system("mkdir output 2> nul"); // Suppress error if dir exists

    std::string cpu_output_path = "output/" + filename_no_ext + "_cpu" + extension;
    std::string gpu_output_path = "output/" + filename_no_ext + "_gpu" + extension;
    std::string avx_output_path = "output/" + filename_no_ext + "_avx" + extension;


    // --- Padding Setup using OpenCV ---
    if (kernel_size % 2 == 0) {
        std::cerr << "Error: Kernel size must be odd for symmetric padding." << std::endl;
        return; // Kernel must be odd
    }
    int pad_rows = (kernel_size - 1) / 2;
    int pad_cols = (kernel_size - 1) / 2;

    cv::Mat padded_image;
    cv::copyMakeBorder(image, padded_image, pad_rows, pad_rows, pad_cols, pad_cols,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0)); // Pad with black

    // Ensure the padded Mat data is contiguous in memory
    if (!padded_image.isContinuous()) {
        padded_image = padded_image.clone();
    }

    int padded_width = padded_image.cols;
    int padded_height = padded_image.rows;
    size_t original_image_size = static_cast<size_t>(width) * height * channels;
    size_t padded_image_size = padded_image.total() * padded_image.elemSize(); // More robust way

    // --- Get raw data pointer from the padded cv::Mat ---
    unsigned char* padded_input_cpu = padded_image.ptr<unsigned char>(0);

    // Allocate other buffers
    unsigned char* padded_input_gpu = new unsigned char[padded_image_size]; // Still need a separate host buffer for GPU memcpy
    unsigned char* cpu_output = new unsigned char[original_image_size];
    unsigned char* gpu_output = new unsigned char[original_image_size];
    unsigned char* cpu_output_avx = new unsigned char[original_image_size];
    unsigned char* kernel = new unsigned char[kernel_size * kernel_size];

    // --- Initialize Kernel ---
    // identity_kernel_init(kernel, kernel_size); // <-- USE THIS FOR DEBUGGING
    box_blur_kernel_init(kernel, kernel_size);
    // box_blur_kernel_init(kernel, kernel_size);
    std::cout << "Kernel initialized" << std::endl;
    // ... (print kernel values if needed) ...

    // --- Copy padded data for GPU ---
    std::memcpy(padded_input_gpu, padded_input_cpu, padded_image_size);

    // --- Save Padded Image (Optional Debug) ---
    cv::imwrite("padded_input_opencv.png", padded_image);
    printf("Padded Dims: %d x %d, Channels: %d\n", padded_width, padded_height, channels);

    // --- Profiling Setup ---
    Profiler::MemoryTracker& mem_tracker = Profiler::MemoryTracker::getInstance();
    mem_tracker.reset();
    // Note: padded_input_cpu points to cv::Mat data, don't track it here
    mem_tracker.record_cpu_allocation(padded_image_size /*padded_gpu*/ + original_image_size * 3 + kernel_size * kernel_size);
    Profiler::MultiApproachComparison perf("Convolution 2D (" + filename_no_ext + ")");

    // --- Run CPU Convolution ---
    Profiler::CPUTimer cpu_timer;
    cpu_timer.start();
    conv2D_cpu(padded_input_cpu, kernel, cpu_output, height, width, kernel_size, kernel_size, channels);
    cpu_timer.stop();
    double cpu_time = cpu_timer.elapsed_milliseconds();
    perf.set_baseline_time(cpu_time);
    perf.record_approach_time("CPU Linear", cpu_time);


    Profiler::CPUTimer cpu_timer_avx;
    cpu_timer_avx.start();
    conv2D_cpu_avx(padded_input_cpu, kernel, cpu_output_avx, height, width, kernel_size, kernel_size, channels);
    cpu_timer_avx.stop();
    double cpu_time_avx = cpu_timer_avx.elapsed_milliseconds();
    perf.record_approach_time("CPU AVX", cpu_time_avx);

    // --- Debug Save CPU Output ---
    // Create a cv::Mat wrapper around the output buffer
    cv::Mat cpu_output_mat_debug(height, width, CV_8UC(channels), cpu_output);
    cv::imwrite("debug_cpu_output.png", cpu_output_mat_debug);

    cv::Mat cpu_output_mat_debug_avx(height, width, CV_8UC(channels), cpu_output_avx);
    cv::imwrite("debug_cpu_output_avx.png", cpu_output_mat_debug_avx);


    // --- Run GPU Convolution ---
    Profiler::KernelTimeTracker::reset();
    conv2D_gpu(padded_input_gpu, kernel, gpu_output, height, width, kernel_size, kernel_size, channels);
    float kernel_time = Profiler::KernelTimeTracker::get_total_kernel_time("conv2D_padded"); // Use the correct name
    perf.record_approach_time("GPU Kernel", kernel_time);


    // --- Save Final Results using OpenCV ---
    cv::Mat cpu_output_mat(height, width, CV_8UC(channels), cpu_output);
    cv::Mat gpu_output_mat(height, width, CV_8UC(channels), gpu_output);
    cv::Mat cpu_output_mat_avx(height, width, CV_8UC(channels), cpu_output_avx);
    cv::imwrite(cpu_output_path, cpu_output_mat);
    cv::imwrite(gpu_output_path, gpu_output_mat);
    cv::imwrite(avx_output_path, cpu_output_mat_avx);

    // --- Print Summary ---
    perf.print_summary();
    Profiler::KernelTimeTracker::print_kernel_times();
    mem_tracker.print_summary();


    // --- Clean up ---
    // delete[] padded_input_cpu; // NO - This points to cv::Mat internal data
    delete[] padded_input_gpu;
    delete[] cpu_output;
    delete[] gpu_output;
    delete[] cpu_output_avx;
    delete[] kernel;
    // cv::Mat objects (image, padded_image, output wrappers) manage their own memory
}

int main(int argc, char** argv) {
    Profiler::print_device_properties();
    // Use OpenCV or standard C++ for directory check/creation if needed
    int kernel_size = 5; 

    if (argc > 1) {
        // ... (argument parsing logic - unchanged) ...
          if (argc > 2 && (std::string(argv[1]) == "-k" || std::string(argv[1]) == "--kernel")) {
             kernel_size = std::stoi(argv[2]);
             if (kernel_size % 2 == 0) {
                 std::cerr << "Error: Kernel size must be odd." << std::endl;
                 return 1;
             }
             std::cout << "Using kernel size: " << kernel_size << std::endl;
             for (int i = 3; i < argc; i++) { 
                process_image(argv[i], kernel_size);
            }
         } else { 
            for (int i = 1; i < argc; i++) {
                process_image(argv[i], kernel_size);
            }
         }
    } else {
        std::string image_dir = "images";
        // Basic check if directory exists - platform specific
        // Could use OpenCV directory functions or C++17 filesystem if linking works
        std::cout << "Processing images in directory: " << image_dir << std::endl;
        // Add logic here to iterate through files in 'images' directory
        // Example using standard C (less robust than filesystem):
        // opendir/readdir/closedir - requires <dirent.h>
        std::vector<std::string> image_files;
        // Placeholder: Manually add image paths if directory iteration is complex now
        image_files.push_back("images/input.jpg"); // Add your test image path
        
        for (const auto& img_path : image_files) {
            process_image(img_path, kernel_size);
        }

        // Replace the filesystem iteration loop with something simpler for now
        // if you don't want to fix the C++17 linking issue
    }

    std::cout << "All images processed successfully!" << std::endl;
    return 0;
}
   
   