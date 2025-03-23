# Vector Addition with CUDA

This project implements vector addition using both CPU and GPU (CUDA) implementations. It serves as a foundational example for learning CUDA programming and performance profiling.

## Overview

The project consists of the following components:
- **vectorAdd_gpu.cu**: CUDA implementation of vector addition.
- **vectorAdd_cpu.cpp**: CPU implementation of vector addition.
- **main.cu**: The main orchestrator that runs tests and profiles both implementations.
- **profiler.h**: A generic profiling module that includes timers, memory usage tracking, and performance comparison.

## Features

- **Performance Profiling**: Measure execution time for both CPU and GPU implementations.
- **Memory Tracking**: Monitor memory usage for CPU and GPU allocations.
- **Robust Testing**: Run multiple test cases with varying vector sizes and display results for verification.

## Getting Started

### Prerequisites

- CUDA Toolkit installed on your machine.
- A compatible NVIDIA GPU.

### Compilation

To compile the project, run:

```bash
nvcc -o vectorAdd vectorAdd_gpu.cu vectorAdd_cpu.cpp main.cu
```

### Running the Program

After compilation, you can run the program using:

```bash
./vectorAdd
```

This will execute the vector addition tests and display the performance metrics for both CPU and GPU implementations.

## Output

The program will display:
- The first 10 elements of the input vectors (A and B) and the result vector (C).
- Performance metrics including CPU time, GPU time, speedup, and memory usage.

## Conclusion

This vector addition example demonstrates the performance benefits of GPU computing over traditional CPU processing. It serves as a simple but effective introduction to CUDA programming concepts including memory allocation, kernel execution, and performance measurement.

Feel free to modify the code to experiment with different vector sizes or optimization techniques!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
