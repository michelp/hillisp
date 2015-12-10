#include "lisp.h"

template<typename T>
__global__ void 
xd_add(T* a, T* b, T* c, size_t size) {
  for (int i = TID; i < size; i += STRIDE)
    c[i] = a[i] + b[i];
}

template<typename T>
__global__ void
 xd_sub(T* a, T* b, T* c, size_t size) {
  for (int i = TID; i < size; i += STRIDE)
    c[i] = a[i] - b[i];
}

template<typename T>
__global__ void
 xd_mul(T* a, T* b, T* c, size_t size) {
  for (int i = TID; i < size; i += STRIDE)
    c[i] = a[i] * b[i];
}

template<typename T>
__global__ void
 xd_div(T* a, T* b, T* c, size_t size) {
  for (int i = TID; i < size; i += STRIDE)
    c[i] = a[i] / b[i];
}

template<typename T>
__global__ void
 xd_fma(T* a, T* b, T* c, size_t size) {
  for (int i = TID; i < size; i += STRIDE)
    c[i] = a[i] * b[i] + c[i];
}


x_any x_add(x_any cell1, x_any cell2) {
  x_any cell;
  if (are_xectors(cell1, cell2)) {
    xectors_align(cell1, cell2);
    cell = new_xector<int64_t>(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_add<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, stream>>>
      (cars<int64_t>(cell1), cars<int64_t>(cell2), cars<int64_t>(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  else if (are_ints(cell1, cell2)) {
    cell = new_cell(NULL, x_int);
    set_val(cell, ival(cell1) + ival(cell2));
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_sub(x_any cell1, x_any cell2) {
  x_any cell;
  if (are_xectors(cell1, cell2)) {
    xectors_align(cell1, cell2);
    cell = new_xector<int64_t>(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_sub<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, stream>>>
      (cars<int64_t>(cell1), cars<int64_t>(cell2), cars<int64_t>(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  else if (are_ints(cell1, cell2)) {
    cell = new_cell(NULL, x_int);
    set_val(cell, ival(cell1) - ival(cell2));
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_mul(x_any cell1, x_any cell2) {
  x_any cell;
  if (are_xectors(cell1, cell2)) {
    xectors_align(cell1, cell2);
    cell = new_xector<int64_t>(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_mul<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, stream>>>
      (cars<int64_t>(cell1), cars<int64_t>(cell2), cars<int64_t>(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  else if (are_ints(cell1, cell2)) {
    cell = new_cell(NULL, x_int);
    set_val(cell, ival(cell1) * ival(cell2));
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_div(x_any cell1, x_any cell2) {
  x_any cell;
  if (are_xectors(cell1, cell2)) {
    xectors_align(cell1, cell2);
    cell = new_xector<int64_t>(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_div<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, stream>>>
      (cars<int64_t>(cell1), cars<int64_t>(cell2), cars<int64_t>(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  else if (are_ints(cell1, cell2)) {
    cell = new_cell(NULL, x_int);
    set_val(cell, ival(cell1) / ival(cell2));
    return cell;
  }
  assert(0);
  return x_nil;
}

x_any x_fma(x_any cell1, x_any cell2, x_any cell3) {
  x_any cell;
  if (are_xectors(cell1, cell2)) {
    xectors_align(cell1, cell2);
    xectors_align(cell1, cell3);
    SYNCS(stream);
    xd_fma<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, stream>>>
      (cars<int64_t>(cell1), cars<int64_t>(cell2), cars<int64_t>(cell3), xector_size(cell1));
    CHECK;
    return cell3;
  }
  else if (are_ints(cell1, cell2)) {
    cell = new_cell(NULL, x_int);
    set_val(cell, ival(cell1) * ival(cell2) + ival(cell3));
    return cell;
  }
  assert(0);
  return x_nil;
}
