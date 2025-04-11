#include "reverseArray.h"

void reverseArray_cpu(float* input, int N){
    for(int i = 0; i < N / 2; i++){
        float temp = input[i];
        input[i] = input[N - i - 1];
        input[N - i - 1] = temp;
    }
}                       