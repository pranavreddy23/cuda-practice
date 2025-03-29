# Softmax Implementation with CUDA

This project implements the softmax function using both CPU (with SIMD optimization) and GPU (CUDA) implementations. It serves as a practical example for learning CUDA programming, SIMD optimization, and performance profiling for neural network activation functions.

## Overview

The project consists of the following components:
- **softmax_gpu.cu**: CUDA implementation of the softmax function.
- **softmax_cpu.cpp**: CPU implementation with SIMD optimization.
- **main.cu**: The main orchestrator that runs tests and profiles both implementations.
- **profiler.cpp**: A generic profiling module that includes timers, memory usage tracking, and performance comparison.

## Features

- **Performance Profiling**: Measure execution time for both CPU and GPU implementations.
- **SIMD Optimization**: CPU implementation leverages vector instructions for parallel computation.
- **Multi-Stage GPU Implementation**: GPU version breaks computation into specialized kernels:
  - Max value reduction
  - Exponentiation computation
  - Sum reduction
  - Final softmax calculation
- **Memory Tracking**: Monitor memory usage for CPU and GPU allocations.
- **Robust Testing**: Run multiple test cases with varying input sizes and display results for verification.

## Getting Started

### Prerequisites

- CUDA Toolkit installed on your machine.
- A compatible NVIDIA GPU.
- CPU with SIMD support for optimal CPU implementation performance.

### Compilation

To compile the project, run:

```bash
nvcc -Xcompiler -mavx -O3 -o softmax softmax_gpu.cu softmax_cpu.cpp main.cu ../common/profiler.cpp
```

### Running the Program

After compilation, you can run the program using:

```bash
./softmax
```

This will execute the softmax function tests and display the performance metrics for both CPU and GPU implementations.

## Output

The program will display:
- Sample input and output values for verification.
- Performance metrics including CPU time, GPU time, speedup, and memory usage.
- Numerical stability analysis ensuring outputs sum to 1.0.

## Technical Implementation

### CPU Implementation

The CPU version uses SIMD instructions for optimization:
- Vector instructions to process multiple elements simultaneously
- Optimized memory access patterns for improved cache performance

### GPU Implementation

The GPU version divides the calculation into four specialized CUDA kernels:
1. `reduce_max_kernel`: Finds the maximum value in the input array.
2. `compute_exp_kernel`: Computes the exponential of each element minus the max value.
3. `reduce_sum_kernel`: Calculates the sum of all exponential values.
4. `softmax_kernel`: Normalizes each element by dividing by the sum.

This multi-kernel approach improves numerical stability and allows for optimization of each computational step.

## Conclusion

This softmax implementation demonstrates the performance benefits of both SIMD optimization on CPU and parallel GPU computing. It serves as an effective introduction to CUDA programming concepts for machine learning operations, including memory allocation, kernel execution, numerical stability considerations, and performance measurement.

Feel free to modify the code to experiment with different input sizes, block configurations, or optimization techniques!


