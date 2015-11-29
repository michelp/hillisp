#include "lisp.h"


__global__ void xd_add(x_any cell1, x_any cell2, x_any cell3, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    int_cars(cell3)[i] = int_cars(cell1)[i] + int_cars(cell2)[i];
  }
}

__global__ void xd_sub(x_any cell1, x_any cell2, x_any cell3, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    int_cars(cell3)[i] = int_cars(cell1)[i] - int_cars(cell2)[i];
  }
}

__global__ void xd_mul(x_any cell1, x_any cell2, x_any cell3, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    int_cars(cell3)[i] = int_cars(cell1)[i] * int_cars(cell2)[i];
  }
}

__global__ void xd_div(x_any cell1, x_any cell2, x_any cell3, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    int_cars(cell3)[i] = int_cars(cell1)[i] / int_cars(cell2)[i];
  }
}

