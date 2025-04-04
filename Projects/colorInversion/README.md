# Color Inversion CUDA Example

This project demonstrates GPU acceleration of image color inversion using CUDA, with CPU SIMD optimizations (AVX2/SSE) for comparison. It's part of a larger collection of CUDA programming examples showcasing parallel computing techniques.

## Overview

The color inversion filter inverts the RGB channels of an image while preserving the alpha channel. This project implements:

1. CPU implementations:
   - Scalar (baseline)
   - SIMD-optimized with AVX2
   - SIMD-optimized with SSE4.1

2. GPU implementation using CUDA

The program automatically selects the best available CPU implementation based on hardware capabilities and compares performance with the GPU version.

## Project Structure

```
.
├── colorInversion.h       # Header file with function declarations
├── colorInversion_cpu.cpp # CPU implementations (scalar, AVX2, SSE)
├── colorInversion_gpu.cu  # CUDA GPU implementation
├── main.cpp               # Main program that processes images and measures performance
├── images/                # Directory for input images
├── output/                # Directory for processed output images
└── common/                # Common utilities
    ├── profiler.cpp       # Performance measurement utilities
    ├── profiler.h         # Header for profiler
    ├── stb_image.h        # Image loading library
    └── stb_image_write.h  # Image writing library
```

## Requirements

- CUDA-capable GPU
- CUDA Toolkit (11.0 or later recommended)
- C++17 compatible compiler
- CPU with AVX2 or SSE4.1 support (for SIMD optimizations)

## Building the Project

Compile the project using NVCC with AVX2 optimization:

```bash
nvcc -Xcompiler "-mavx2 -O3" -o colorInversion colorInversion_gpu.cu colorInversion_cpu.cpp main.cpp ../common/profiler.cpp
```


## Usage

### Process a single image:

```bash
./colorInversion image.png
```

### Process multiple specific images:

```bash
./colorInversion image1.jpg image2.png image3.bmp
```

### Process all images in the "images" directory:

```bash
./colorInversion
```

Output images will be saved in the "output" directory with "_cpu" and "_gpu" suffixes.

## Implementation Details

### CPU Implementation

The CPU implementation leverages SIMD instructions for parallelism:

- **Scalar**: Basic implementation processing one pixel at a time
- **AVX2**: Processes 8 pixels (32 bytes) at once using 256-bit vectors
- **SSE4.1**: Processes 4 pixels (16 bytes) at once using 128-bit vectors

The appropriate implementation is selected at runtime based on CPU capabilities.

### GPU Implementation

The CUDA implementation parallelizes the color inversion across thousands of threads, with each thread handling one or more pixels. The implementation optimizes memory access patterns and utilizes GPU shared memory where beneficial.

## Performance Metrics

The program automatically benchmarks and reports:

- Execution time for both CPU and GPU implementations
- Speedup factor of GPU vs. CPU
- Memory usage
- Detailed kernel timing breakdown for GPU execution

## Key Optimizations

1. **SIMD Vectorization**: Utilizes AVX2/SSE instructions to process multiple pixels in parallel on the CPU
2. **Memory Access Patterns**: Optimized for coalesced memory access on the GPU
3. **Automatic CPU Feature Detection**: Selects the most efficient CPU implementation based on hardware capabilities


## Acknowledgments

- stb_image and stb_image_write by Sean Barrett
