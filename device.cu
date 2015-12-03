#include "lisp.h"

__global__ void xd_add_xint64(x_any cell1, x_any cell2, x_any cell3, int size) {
  if (TID < size)
    int64_cars(cell3)[TID] = int64_cars(cell1)[TID] + int64_cars(cell2)[TID];
}

__global__ void xd_sub(x_any cell1, x_any cell2, x_any cell3, int size) {
  if (TID < size)
    int64_cars(cell3)[TID] = int64_cars(cell1)[TID] - int64_cars(cell2)[TID];
}

__global__ void xd_mul(x_any cell1, x_any cell2, x_any cell3, int size) {
  if (TID < size)
    int64_cars(cell3)[TID] = int64_cars(cell1)[TID] * int64_cars(cell2)[TID];
}

__global__ void xd_div(x_any cell1, x_any cell2, x_any cell3, int size) {
  if (TID < size) 
    int64_cars(cell3)[TID] = int64_cars(cell1)[TID] / int64_cars(cell2)[TID];
}

__global__ void xd_zeros(x_any cell, int size) {
  if (TID < size)
    int64_cars(cell)[TID] = 0;
}

__global__ void xd_ones(x_any cell, int size) {
  if (TID < size)
    int64_cars(cell)[TID] = 1;
}
