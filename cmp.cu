#include "lisp.h"

template<typename T>
__global__ void
 xd_eq(const T* __restrict__ cell1, const T* __restrict__ cell2, T* __restrict__ cell3, const size_t size) {
  int i = TID;
  while (i < size) {
    cell3[i] = (T)(cell1[i] == cell2[i]);
    i += STRIDE;
  }
}

x_any x_eq(x_any cell1, x_any cell2) {
  x_any cell;
  if (are_ints(cell1, cell2)) {
    if (ival(cell1) == ival(cell2))
      return x_env.x_true;
  }
  else if (are_xectors(cell1, cell2)) {
    xectors_align(cell1, cell2);
    cell = new_xector<int64_t>(NULL, xector_size(cell1));
    SYNCS(x_env.stream);
    xd_eq<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(cell1), cars<int64_t>(cell2), cars<int64_t>(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  else if (are_atoms(cell1, cell2)) {
    if (strcmp(sval(cell1), sval(cell2)) == 0)
      return x_env.x_true;
  }
  else if (are_pairs(cell1, cell2)) {
    do {
      if (x_eq(car(cell1), car(cell2)) == x_env.x_nil)
        return x_env.x_nil;
      cell1 = cdr(cell1);
      cell2 = cdr(cell2);
    } while (are_pairs(cell1, cell2));
    if (x_eq(cell1, cell2) != x_env.x_nil)
      return x_env.x_true;
  }
  return x_env.x_nil;
}

x_any x_neq(x_any cell1, x_any cell2) {
  if (x_eq(cell1, cell2) == x_env.x_true)
    return x_env.x_nil;
  return x_env.x_true;
}

x_any x_gt(x_any cell1, x_any cell2) {
  if (are_ints(cell1, cell2)) {
    if (ival(cell1) > ival(cell2))
      return x_env.x_true;
  }
  else
    if (strcmp(sval(cell1), sval(cell2)) > 0)
      return x_env.x_true;
  return x_env.x_nil;
}

x_any x_lt(x_any cell1, x_any cell2) {
  if (are_ints(cell1, cell2)) {
    if (ival(cell1) < ival(cell2))
      return x_env.x_true;
  }
  else
    if (strcmp(sval(cell1), sval(cell2)) < 0)
      return x_env.x_true;
  return x_env.x_nil;
}
