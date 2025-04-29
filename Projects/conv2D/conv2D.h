#ifndef CONV2D_H
#define CONV2D_H

extern "C" void conv2D_gpu(unsigned char* input, unsigned char* kernel, unsigned char* output,
               int input_rows, int input_cols, int kernel_rows, int kernel_cols, int channels);   
void conv2D_cpu(unsigned char* input, unsigned char* kernel, unsigned char* output,
               int input_rows, int input_cols, int kernel_rows, int kernel_cols, int channels);   
void conv2D_cpu_avx(unsigned char* input, unsigned char* kernel, unsigned char* output,
               int input_rows, int input_cols, int kernel_rows, int kernel_cols, int channels);   

#ifdef __CUDACC__
__global__ void convolutionKernelPadded (unsigned char* input, unsigned char* kernel, unsigned char* output,
               int original_rows, int original_cols, int kernel_rows, int kernel_cols, int channels);   
#endif

#endif
