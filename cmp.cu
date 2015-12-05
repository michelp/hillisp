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
    if (int64_car(cell1) == int64_car(cell2))
      return x_true;
  }
  else if (are_xectors(cell1, cell2)) {
    xectors_align(cell1, cell2);
    cell = new_xector(NULL, xector_size(cell1));
    SYNCS(stream);
    xd_eq<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, stream>>>
      (int64_cars(cell1), int64_cars(cell2), int64_cars(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  else if (is_atom(cell1) && is_atom(cell2)) {
    if (strcmp(cell1->name, cell2->name) == 0)
      return x_true;
  }
  else if (is_pair(cell1) && is_pair(cell2)) {
    do {
      if (x_eq(car(cell1), car(cell2)) == x_nil)
        return x_nil;
      cell1 = cdr(cell1);
      cell2 = cdr(cell2);
    } while (is_pair(cell1) && is_pair(cell2));
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
    if (int64_car(cell1) > int64_car(cell2))
      return x_true;
  }
  else
    if (strcmp(cell1->name, cell2->name) > 0)
      return x_true;
  return x_nil;
}

x_any x_lt(x_any cell1, x_any cell2) {
  if (are_ints(cell1, cell2)) {
    if (int64_car(cell1) < int64_car(cell2))
      return x_true;
  }
  else
    if (strcmp(cell1->name, cell2->name) < 0)
      return x_true;
  return x_nil;
}
