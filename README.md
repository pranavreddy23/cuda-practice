# CUDA Practice

Welcome to the **CUDA Practice** repository! This project is a collection of exercises aimed at exploring and mastering GPU programming using **CUDA** (Compute Unified Device Architecture). This is an ongoing project that contains progressively more complex CUDA exercises to help understand core concepts of parallelism, memory management, kernel execution, and performance profiling on GPUs.

## Repository Structure

This repository is organized with a Projects directory containing individual CUDA programming exercises:

```
cuda-practice/
├── Projects/
│   ├── common/
│   │   ├── [shared utility files]
│   ├── vectorAdd/
│   │   ├── [implementation files]
│   │   ├── README.md
│   ├── [future-project-1]/
│   │   ├── [implementation files]
│   │   ├── README.md
│   └── ...
```

Each project directory contains:
- Implementation files (CUDA, C/C++)
- A dedicated README.md with specific instructions, explanations, and compilation commands
- The `common` directory contains shared utilities that may be used across multiple projects

## Project Overview

As I progress in my CUDA programming journey, this repository will expand with new projects covering various aspects of GPU programming:

### **Current and Planned Projects**

- **Vector Addition**: A fundamental exercise demonstrating basic CUDA concepts
- **Matrix Operations**: Matrix multiplication and transformations
- **Image Processing**: Filters, transformations, and convolutions
- **Reduction Operations**: Sum, min/max, and other reduction algorithms
- **Sorting Algorithms**: Parallel implementations of sorting techniques
- **Physics Simulations**: N-body problems and particle systems

Each project will focus on specific CUDA concepts and optimization techniques.

---

## Prerequisites

Before you begin, make sure you have the following tools installed:

1. **CUDA Toolkit**: Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) to access nvcc (the CUDA compiler) and other essential tools.
2. **NVIDIA GPU**: Ensure you have a compatible NVIDIA GPU to execute the CUDA programs.
3. **Profiling Tools**: You'll need Nsight Systems (nsys) for profiling and analyzing the performance of your CUDA programs.

---

## Getting Started

### **1. Clone the Repository**

First, clone the repository to your local machine:

```bash
git clone https://github.com/pranavreddy23/cuda-practice.git
cd cuda-practice
```

### **2. Navigate to the Projects Directory**

```bash
cd Projects
```

### **3. Choose a Project**

Navigate to the project you want to explore:

```bash
cd vectorAdd  # or any other project directory
```

### **4. Follow Project-Specific Instructions**

Each project directory contains its own README.md with specific instructions for:
- Compiling the code
- Running the program
- Profiling the execution
- Understanding the concepts demonstrated

### **5. Profiling CUDA Programs**

For performance analysis, use **Nsight Systems** (`nsys`) to profile your programs:

```bash
nsys nvprof ./[executable_name]
```

This will generate detailed performance reports with insights into GPU usage, kernel execution time, and memory transfer patterns.

## Learning Path

I recommend working through the projects in order, as they build upon concepts introduced in previous projects. Each project will introduce new CUDA programming concepts and optimization techniques.

## Contributions

Feel free to suggest improvements or additional projects that might be helpful in my CUDA learning journey!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
