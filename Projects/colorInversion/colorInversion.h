#ifndef COLORINVERSION_H
#define COLORINVERSION_H

void colorInversion_gpu(unsigned char* image, int width, int height);

void colorInversion_cpu(unsigned char* image, int width, int height);

#ifdef __CUDACC__
__global__ void invert_kernel(unsigned char* image, int width, int height);
#endif

#endif // COLORINVERSION_H
