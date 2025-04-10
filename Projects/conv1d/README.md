# 1D Convolution CUDA Implementation

This project demonstrates the implementation of 1D convolution using CUDA, comparing the performance between CPU and GPU implementations.

## Overview

1D convolution is a fundamental operation in signal processing, used for tasks such as filtering, smoothing, and feature extraction. This implementation showcases how to accelerate 1D convolution operations using CUDA parallel processing on the GPU, achieving significant performance improvements for large datasets.

## Features

- CUDA-accelerated 1D convolution with tiled shared memory approach
- CPU implementation for performance comparison
- Automatic performance testing across various data sizes
- Boundary handling with zero padding

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit (version 10.0 or higher recommended)
- C++ compiler compatible with your CUDA version

## Building the Project

To build the project, use the following command:

```bash
nvcc -O3 main.cu conv1d_gpu.cu conv1d_cpu.cpp ../common/profiler.cpp -o conv1D
```

## Running the Program

Execute the compiled binary:

```bash
./conv1D
```

The program will run tests with different data sizes (ranging from 1,000 to 1,000,000 elements) and compare the performance between CPU and GPU implementations.

## Implementation Details

### CPU Implementation

The CPU implementation includes two versions:
- Standard 1D convolution without padding
- 1D convolution with zero padding to maintain the input size

### GPU Implementation

The GPU implementation leverages several optimization techniques:
- Shared memory to reduce global memory access latency
- Tiled approach with halo regions for efficient processing
- Block and thread organization for optimal parallelism
- Zero padding for boundary handling

### Test Data

The program generates synthetic test data consisting of sine waves with added noise. A Gaussian filter is applied to smooth the data, demonstrating a common use case for 1D convolution in signal processing applications.

## Expected Output

For each test size, the program outputs:
- Verification result (PASSED/FAILED)
- CPU execution time
- GPU execution time
- Speedup factor (CPU time / GPU time)

Example output:
```
=== Testing with data size: 10000 ===
Verification: PASSED
CPU time: 0.456 ms
GPU time: 0.089 ms
Speedup: 5.12x
```

## Performance Considerations

- The GPU implementation shows significant speedup for larger data sizes
- For small data sizes, the overhead of data transfer between CPU and GPU may reduce the performance advantage
- The speedup factor typically increases with larger input sizes due to better GPU utilization
- The optimal block size may vary depending on your GPU architecture

## Contributing

Contributions to improve the implementation or add new features are welcome. Please feel free to submit a pull request or open an issue to discuss potential improvements.


