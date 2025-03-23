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

To compile the project, navigate to the `Projects/vectorAdd` directory and run:

bash
nvcc -o vectorAdd vectorAdd_gpu.cu vectorAdd_cpu.cpp main.cu

### Running the Program

After compilation, you can run the program using:

bash
./vectorAdd


This will execute the vector addition tests and display the performance metrics for both CPU and GPU implementations.

## Output

The program will display:

- The first 10 elements of the input vectors (A and B) and the result vector (C).
- Performance metrics including CPU time, GPU time, speedup, and memory usage.

## Conclusion

This project serves as a starting point for exploring CUDA programming and performance profiling. You can extend it by adding more complex operations or additional profiling metrics.

Feel free to contribute or modify the project as you see fit!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

