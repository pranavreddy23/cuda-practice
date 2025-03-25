# Matrix Multiplication: CPU vs GPU Implementation

## Overview

This project provides a comprehensive implementation of matrix multiplication, demonstrating both CPU and GPU (CUDA) approaches. It offers a performance comparison between traditional CPU-based computation and GPU-accelerated matrix multiplication.

## Project Structure

```
project/
│
├── matmul/
│   ├── main.cu          # Test harness for matrix multiplication
│   ├── matmul.h         # Header file with function declarations
│   ├── matmul_cpu.cpp   # CPU implementation
│   └── matmul_gpu.cu    # CUDA GPU implementation
│
└── common/
    └── profiler.h       # Performance profiling utilities
```

## Features

- Parallel matrix multiplication using CUDA
- Performance benchmarking
- Comprehensive test suite covering various matrix sizes
- Detailed timing and memory usage statistics

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- C++ compiler
- CMake (optional, for build system)

## Implementation Details

### CPU Implementation

The CPU version uses a standard triple-nested loop approach, computing matrix multiplication sequentially on a single core.

### GPU Implementation

The CUDA implementation leverages parallel computing:
- Uses 2D grid of thread blocks
- Each thread computes a single output matrix element
- Optimized memory access patterns
- Handles various matrix dimensions

## Performance Metrics

- CPU execution time
- GPU total execution time
- GPU kernel execution time
- Memory transfer overhead
- Performance speedup comparisons

## Test Cases

Matrix multiplication tests include:
- Small matrices (16x16, 32x32)
- Medium matrices (128x128, 256x256)
- Large matrices (512x512, 1024x1024)
- Non-square matrices
- Edge cases (1x1 matrices, dot products)

## Building the Project

### Compilation

```bash
# Basic compilation
nvcc -o matmul main.cu matmul_gpu.cu matmul_cpu.cpp

# With optimizations (recommended)
nvcc -O3 -o matmul main.cu matmul_gpu.cu matmul_cpu.cpp
```

### Running Tests

```bash
# Execute the matrix multiplication benchmark
./matmul
```

## Performance Tips

- Ensure input matrices are appropriately sized for GPU
- Use pinned (page-locked) memory for faster transfers
- Experiment with different block sizes
- Profile and optimize kernel configuration

## Potential Improvements

- Implement more advanced tiling techniques
- Add support for double-precision matrices
- Create adaptive block size selection
- Integrate with cuBLAS for comparison

## License

[Insert your preferred open-source license here]

## Contributing

Contributions are welcome! Please submit pull requests or open issues to suggest improvements or report bugs.

## Acknowledgments

- NVIDIA CUDA Programming Guide
- Academic and research computing resources
```
