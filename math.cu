#include "lisp.h"

template<typename T>
__global__ void 
xd_add(T* cell1, T* cell2, T* cell3, size_t size) {
  int i = TID;
  while (i < size) {
    cell3[i] = cell1[i] + cell2[i];
    i += STRIDE;
  }
}

template<typename T>
__global__ void
 xd_sub(T* cell1, T* cell2, T* cell3, size_t size) {
  int i = TID;
  while (i < size) {
    cell3[i] = cell1[i] - cell2[i];
    i += STRIDE;
  }
}

template<typename T>
__global__ void xd_mul(T* cell1, T* cell2, T* cell3, size_t size) {
  int i = TID;
  while (i < size) {
    cell3[i] = cell1[i] * cell2[i];
    i += STRIDE;
  }
}

template<typename T>
__global__ void xd_div(T* cell1, T* cell2, T* cell3, size_t size) {
  int i = TID;
  while (i < size) {
    cell3[i] = cell1[i] / cell2[i];
    i += STRIDE;
  }
}

template<typename T>
__global__ void xd_fma(T* cell1, T* cell2, T* cell3, size_t size) {
  int i = TID;
  while (i < size) {
    cell3[i] = cell1[i] * cell2[i] + cell3[i];
    i += STRIDE;
  }
}

x_any x_add(x_any cell1, x_any cell2) {
  x_any cell;
  if (are_ints(cell1, cell2)) {
    cell = new_cell(NULL, x_int);
    set_car(cell, int64_car(cell1) + int64_car(cell2));
    return cell;
  }
  else if (are_xectors(cell1, cell2)) {
    xectors_align(cell1, cell2)
    cell = new_xector(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_add<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, stream>>>
      (int64_cars(cell1), int64_cars(cell2), int64_cars(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_sub(x_any cell1, x_any cell2) {
  x_any cell;
  if (are_ints(cell1, cell2)) {
    cell = new_cell(NULL, x_int);
    set_car(cell, int64_car(cell1) - int64_car(cell2));
    return cell;
  }
  else if (are_xectors(cell1, cell2)) {
    xectors_align(cell1, cell2)
    cell = new_xector(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_sub<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, stream>>>
      (int64_cars(cell1), int64_cars(cell2), int64_cars(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_mul(x_any cell1, x_any cell2) {
  x_any cell;
  if (are_ints(cell1, cell2)) {
    cell = new_cell(NULL, x_int);
    set_car(cell, int64_car(cell1) * int64_car(cell2));
    return cell;
  }
  else if (are_xectors(cell1, cell2)) {
    xectors_align(cell1, cell2)
    cell = new_xector(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_mul<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, stream>>>
      (int64_cars(cell1), int64_cars(cell2), int64_cars(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_div(x_any cell1, x_any cell2) {
  x_any cell;
  if (are_ints(cell1, cell2)) {
    cell = new_cell(NULL, x_int);
    set_car(cell, int64_car(cell1) / int64_car(cell2));
    return cell;
  }
  else if (are_xectors(cell1, cell2)) {
    xectors_align(cell1, cell2)
    cell = new_xector(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_div<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, stream>>>
      (int64_cars(cell1), int64_cars(cell2), int64_cars(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_fma(x_any cell1, x_any cell2, x_any cell3) {
  x_any cell;
  if (are_ints(cell1, cell2)) {
    cell = new_cell(NULL, x_int);
    set_car(cell, int64_car(cell1) * int64_car(cell2) + int64_car(cell3));
    return cell;
  }
  else if (are_xectors(cell1, cell2)) {
    SYNCS(stream);
    xectors_align(cell1, cell2)
    xectors_align(cell1, cell3)
    xd_fma<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, stream>>>
      (int64_cars(cell1), int64_cars(cell2), int64_cars(cell3), xector_size(cell1));
    CHECK;
    return cell3;
  }
  assert(0);
  return x_nil;
}
