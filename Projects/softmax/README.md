# Softmax Implementation with CUDA

This project implements the softmax function using both CPU (with SIMD/AVX optimization) and GPU (CUDA) implementations. It serves as a practical example for learning CUDA programming, SIMD optimization, and performance profiling for neural network activation functions.

## Overview

The project consists of the following components:
- **softmax_gpu.cu**: CUDA implementation of the softmax function with multiple specialized kernels.
- **softmax_cpu.cpp**: CPU implementation with AVX SIMD optimization for improved performance.
- **main.cu**: The main orchestrator that runs tests and profiles both implementations.
- **profiler.cpp**: A generic profiling module that includes timers, memory usage tracking, and performance comparison.

## Features

- **Performance Profiling**: Measure execution time for both CPU and GPU implementations.
- **SIMD Optimization**: CPU implementation leverages AVX instructions (`__m256`) to process 8 float values simultaneously.
- **Multi-Stage GPU Implementation**: GPU version breaks computation into specialized kernels:
  - Max value reduction
  - Exponentiation computation
  - Sum reduction
  - Final softmax calculation
- **Memory Tracking**: Monitor memory usage for CPU and GPU allocations.
- **Numerical Stability**: Both implementations use the max-subtraction technique to prevent overflow in exponential calculations.
- **Robust Testing**: Run multiple test cases with varying input sizes and display results for verification.

## Getting Started

### Prerequisites

- CUDA Toolkit installed on your machine.
- A compatible NVIDIA GPU.
- CPU with AVX support for optimal CPU implementation performance.

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

### CPU Implementation (SIMD Optimized)

The CPU implementation uses AVX instructions for SIMD optimization with a three-step approach:

1. **Maximum Finding**:
   - Uses `_mm256_max_ps` to process 8 elements at a time
   - Reduces the SIMD vector to find the global maximum
   - Falls back to scalar code for arrays smaller than 8 elements or remainder elements

2. **Exponentiation**:
   - Subtracts the maximum value from each element using `_mm256_sub_ps`
   - Calculates exponentials (currently using scalar `std::exp` for accuracy)
   - Simultaneously accumulates the sum of all exponentials

3. **Normalization**:
   - Computes the inverse of the sum (1.0/sum)
   - Uses `_mm256_mul_ps` to multiply each exponential result by this inverse
   - Produces the final normalized softmax values

The implementation automatically falls back to scalar code for small arrays (< 8 elements), ensuring compatibility across different input sizes.

### GPU Implementation

The GPU version divides the calculation into four specialized CUDA kernels:
1. `reduce_max_kernel`: Finds the maximum value in the input array.
2. `compute_exp_kernel`: Computes the exponential of each element minus the max value.
3. `reduce_sum_kernel`: Calculates the sum of all exponential values.
4. `softmax_kernel`: Normalizes each element by dividing by the sum.

This multi-kernel approach improves numerical stability and allows for optimization of each computational step, leveraging the GPU's parallel architecture for significant performance gains with large datasets.

## Performance Considerations

- **CPU Implementation**: Performs well for small to medium-sized inputs, especially when data fits in L1/L2 cache.
- **GPU Implementation**: Excels with larger inputs where parallelism can be fully utilized.
- **Memory Transfer**: For small inputs, the overhead of data transfer to and from the GPU may offset the computational advantages.
- **Batch Processing**: Both implementations can be extended to efficiently handle batched inputs for neural network applications.

## Conclusion

This softmax implementation demonstrates the performance benefits of both SIMD optimization on CPU and parallel GPU computing. It serves as an effective introduction to CUDA programming concepts for machine learning operations, including memory allocation, kernel execution, numerical stability considerations, and performance measurement.

Feel free to modify the code to experiment with different input sizes, block configurations, or optimization techniques!

