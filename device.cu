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

__global__ void xd_zeros(x_any cell, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    int_cars(cell)[i] = 0;
  }
}

__global__ void xd_ones(x_any cell, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    int_cars(cell)[i] = 1;
  }
}

__device__ x_any xd_add2(x_any cell1, x_any cell2) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  x_any result;
  size_t size;
  if (i == 0) {
    if (xector_size(cell1) != xector_size(cell2))
      assert(0);
    size = xector_size(cell1);
  }
  __syncthreads();
  result = new_xector(NULL);
  if (i < size) {
    int_cars(result)[i] = int_cars(cell1)[i] + int_cars(cell2)[i];
  }
  return result;
}

__device__ void xd_sub2(x_any cell1, x_any cell2, x_any cell3, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    int_cars(cell3)[i] = int_cars(cell1)[i] - int_cars(cell2)[i];
  }
}

__device__ void xd_mul2(x_any cell1, x_any cell2, x_any cell3, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    int_cars(cell3)[i] = int_cars(cell1)[i] * int_cars(cell2)[i];
  }
}

__device__ void xd_div2(x_any cell1, x_any cell2, x_any cell3, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    int_cars(cell3)[i] = int_cars(cell1)[i] / int_cars(cell2)[i];
  }
}


__device__ x_any xd_apply(x_any cell, x_any args) {
  if (is_builtin(cell)) {
    if (is_fn0(cell))
      return ((x_fn0_t)car(cell))();
    else if (is_fn1(cell))
      return ((x_fn1_t)car(cell))(car(args));
    else if (is_fn2(cell))
      return ((x_fn2_t)car(cell))(car(args), cadr(args));
    else if (is_fn3(cell))
      return ((x_fn3_t)car(cell))(car(args), cadr(args), caddr(args));
    else
      assert(0);
  }
  else
    assert(0);
  return x_nil;
}

__global__ void xd_eval(x_any cell, x_any result) {
  if (is_atom(cell))
    copy_cell(cell, result);
  else if (is_pair(cell) && (is_func(car(cell))))
    copy_cell(xd_apply(car(cell), cdr(cell)), result);
}
