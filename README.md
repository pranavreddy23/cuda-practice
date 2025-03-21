
# CUDA Practice

Welcome to the **CUDA Practice** repository! This project is a collection of exercises aimed at exploring and mastering GPU programming using **CUDA** (Compute Unified Device Architecture). This is an ongoing project where you will find simple CUDA exercises, starting with the vectorAdd program and adding more over time. The goal is to understand core concepts of parallelism, memory management, kernel execution, and performance profiling on GPUs.

## Project Overview

The first exercise, vectorAdd, is a simple program that performs **element-wise addition of two vectors** using CUDA. As the project progresses, more complex exercises will be added, focusing on advanced CUDA features such as memory management, optimization, and parallel algorithms.

### **Upcoming Exercises**
- New CUDA exercises will be progressively added, covering topics like kernel optimization, memory management, parallel algorithms, and performance tuning.

---

## Prerequisites

Before you begin, make sure you have the following tools installed:

1. **CUDA Toolkit**: Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) to access nvcc (the CUDA compiler) and other essential tools.
2. **NVIDIA GPU**: Ensure you have a compatible NVIDIA GPU to execute the CUDA programs.
3. **Profiling Tools**: You'll need nvprof or nsys to profile and analyze the performance of your CUDA programs.

---

## Getting Started

### **1. Clone the Repository**

First, clone the repository to your local machine:

```bash
git clone https://github.com/pranavreddy23/cuda-practice.git
cd cuda-practice
```

### **2. Compile the Program**

Once you have cloned the repository, navigate to the folder containing your CUDA files. To compile the program (e.g., `vectorAdd.cu`), use the following command:

```bash
nvcc vectorAdd.cu -o vectorAdd
```

This will generate an executable named `vectorAdd`.

### **3. Run the Program**

After compiling, run the executable:

```bash
./vectorAdd
```

This will execute the `vectorAdd` CUDA program and print the result of adding two vectors.

### **4. Profiling the Program**

For performance analysis, use **Nsight Systems** (`nsys`) or **nvprof** to profile your program and understand how well it is utilizing the GPU.But nvprof is not supported on devices with compute capability 8.0 and higher.



* Using `nsys`:

```bash
nsys nvprof ./vectorAdd
```

These commands will run the program and generate performance reports that give you insights into GPU usage, kernel execution time, and memory transfer details.
