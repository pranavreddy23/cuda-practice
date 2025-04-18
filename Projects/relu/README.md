# ReLU (Rectified Linear Unit) GPU Implementation

This project implements and benchmarks the ReLU activation function using different approaches:

1. CPU (sequential)
2. CPU with AVX vectorization
3. GPU implementation with CUDA
   - Standard implementation
   - Vectorized (float4) implementation

## Overview

ReLU is a simple yet widely used activation function in deep learning, defined as:

```
f(x) = max(0, x)
```

This implementation compares the performance of this operation across different hardware and optimization techniques.

## Compilation

Compile the project using NVIDIA's nvcc compiler with AVX2 support:

```bash
nvcc -Xcompiler -mavx2 -O3 -o relu main.cu relu_cpu.cpp relu_gpu.cu ../common/profiler.cpp
```

## Usage

Run the compiled executable:

```bash
./relu
```

This will run the ReLU operation on matrices of increasing sizes and report performance metrics.

## Implementation Details

### CPU Implementation
- Basic sequential implementation iterating through each element
- AVX vectorized implementation processing 8 floats per instruction

### GPU Implementation
- CUDA kernel with coalesced memory access
- Optimized vectorized version using float4 data type for 4x throughput
- Dynamic selection between standard and vectorized kernels based on data size

## Performance Analysis

The program measures and reports:
- Execution time for each approach
- Speedup relative to CPU baseline
- Memory usage (host and device)
- Result verification against reference CPU implementation

## Hardware Requirements

- CUDA-capable NVIDIA GPU
- CPU with AVX2 support (for AVX implementation)

