#include "lisp.h"

template<typename T>
__global__ void
 xd_eq(T *cell1, T* cell2, T* cell3, size_t size) {
  int i = TID;
  while (i < size) {
    cell3[i] = (T)(cell1[i] == cell2[i]);
    i += STRIDE;
  }
}

x_any x_eq(x_any cell1, x_any cell2) {
  x_any cell;
  if (are_ints(cell1, cell2)) {
    if (carr<int64_t>(cell1) == carr<int64_t>(cell2))
      return x_true;
  }
  else if (are_xectors(cell1, cell2)) {
    xectors_align(cell1, cell2);
    cell = new_xector(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_eq<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, stream>>>
      (carrs<int64_t>(cell1), carrs<int64_t>(cell2), carrs<int64_t>(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  else if (are_atoms(cell1, cell2)) {
    if (strcmp(cell1->name, cell2->name) == 0)
      return x_true;
  }
  else if (are_pairs(cell1, cell2)) {
    do {
      if (x_eq(carr<x_any>(cell1), carr<x_any>(cell2)) == x_nil)
        return x_nil;
      cell1 = cdrr<x_any>(cell1);
      cell2 = cdrr<x_any>(cell2);
    } while (are_pairs(cell1, cell2));
    if (x_eq(cell1, cell2) != x_nil)
      return x_true;
  }
  return x_nil;
}

x_any x_neq(x_any cell1, x_any cell2) {
  if (x_eq(cell1, cell2) == x_true)
    return x_nil;
  return x_true;
}

x_any x_gt(x_any cell1, x_any cell2) {
  if (are_ints(cell1, cell2)) {
    if (carr<int64_t>(cell1) > carr<int64_t>(cell2))
      return x_true;
  }
  else
    if (strcmp(cell1->name, cell2->name) > 0)
      return x_true;
  return x_nil;
}

x_any x_lt(x_any cell1, x_any cell2) {
  if (are_ints(cell1, cell2)) {
    if (carr<int64_t>(cell1) < carr<int64_t>(cell2))
      return x_true;
  }
  else
    if (strcmp(cell1->name, cell2->name) < 0)
      return x_true;
  return x_nil;
}
