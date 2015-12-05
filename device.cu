#include <curand.h>
#include "lisp.h"

__global__ void 
xd_add_xint64(int64_t* cell1, int64_t* cell2, int64_t* cell3, size_t size) {
  if (TID < size)
    cell3[TID] = cell1[TID] + cell2[TID];
}

__global__ void
 xd_sub_xint64(int64_t* cell1, int64_t* cell2, int64_t* cell3, size_t size) {
  if (TID < size)
    cell3[TID] = cell1[TID] - cell2[TID];
}

__global__ void xd_mul_xint64(int64_t* cell1, int64_t* cell2, int64_t* cell3, size_t size) {
  if (TID < size)
    cell3[TID] = cell1[TID] * cell2[TID];
}

__global__ void xd_div_xint64(int64_t* cell1, int64_t* cell2, int64_t* cell3, size_t size) {
  if (TID < size) 
    cell3[TID] = cell1[TID] / cell2[TID];
}

__global__ void xd_fma_xint64(int64_t* cell1, int64_t* cell2, int64_t* cell3, size_t size) {
  if (TID < size) 
    cell3[TID] = cell1[TID] * cell2[TID] + cell3[TID];
}

__global__ void
 xd_eq_xint64(int64_t* cell1, int64_t* cell2, int64_t* cell3, size_t size) {
  if (TID < size) 
    cell3[TID] = (int64_t)(cell1[TID] == cell2[TID]);
}

__global__ void
 xd_fill_xint64(int64_t *cars, int64_t val, size_t size) {
  if (TID < size)
    cars[TID] = val;
}

__global__ void
 xd_rand_xint64(int64_t *cars, int64_t val, size_t size) {
  if (TID < size)
    cars[TID] = val;
}

__global__ void
 xd_all_xint64(int64_t* cell1, int *result, size_t size) {
  if (*result == size)
    if (TID < size)
      if (!cell1[TID])
        atomicSub(result, 1);
  __syncthreads();
}

__global__ void
 xd_any_xint64(int64_t* cell1, int *result, size_t size) {
  if (*result == 0)
    if (TID < size)
      if (cell1[TID])
        atomicAdd(result, 1);
  __syncthreads();
}

