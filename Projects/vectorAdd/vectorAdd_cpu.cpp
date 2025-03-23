#include "vectorAdd.h"

// CPU implementation of vector addition
void vectorAdd_cpu(const std::vector<int>& a, const std::vector<int>& b,
                  std::vector<int>& c, int N) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}
