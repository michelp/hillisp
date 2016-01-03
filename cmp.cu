#include "lisp.h"

template<typename T>
__global__ void
 xd_eq(const T* __restrict__ cell1, const T* __restrict__ cell2, int64_t* __restrict__ cell3, const size_t size) {
  int i = TID;
  while (i < size) {
    cell3[i] = (int64_t)(cell1[i] == cell2[i]);
    i += STRIDE;
  }
}

x_any x_eq(x_any cell1, x_any cell2) {
  x_any cell;
  if (cell1 == cell2)
    return x_env.true_;
  else if (are_symbols(cell1, cell2)) {
    if (strcmp(sval(cell1), sval(cell2)) == 0)
      return x_env.true_;
  }
  else if (are_ints(cell1, cell2)) {
    if (ival(cell1) == ival(cell2))
      return x_env.true_;
  }
  else if (are_doubles(cell1, cell2)) {
    if (fval(cell1) == fval(cell2))
      return x_env.true_;
  }
  else if (are_ixectors(cell1, cell2)) {
    assert_xectors_align(cell1, cell2);
    cell = new_ixector(xector_size(cell1));
    SYNCS(x_env.stream);
    xd_eq<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(cell1), cars<int64_t>(cell2), cars<int64_t>(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  else if (are_dxectors(cell1, cell2)) {
    assert_xectors_align(cell1, cell2);
    cell = new_ixector(xector_size(cell1));
    SYNCS(x_env.stream);
    xd_eq<double><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<double>(cell1), cars<double>(cell2), cars<int64_t>(cell), xector_size(cell1));
    CHECK;
    return cell;
  }
  else if (are_strs(cell1, cell2)) {
    if (strcmp(sval(cell1), sval(cell2)) == 0)
      return x_env.true_;
  }
  else if (are_pairs(cell1, cell2)) {
    do {
      if (x_eq(car(cell1), car(cell2)) == x_env.nil)
        return x_env.nil;
      cell1 = cdr(cell1);
      cell2 = cdr(cell2);
    } while (are_pairs(cell1, cell2));
    if (x_eq(cell1, cell2) != x_env.nil)
      return x_env.true_;
  }
  return x_env.nil;
}

x_any x_neq(x_any cell1, x_any cell2) {
  if (x_eq(cell1, cell2) == x_env.true_)
    return x_env.nil;
  return x_env.true_;
}

x_any x_gt(x_any cell1, x_any cell2) {
  if (are_ints(cell1, cell2)) {
    if (ival(cell1) > ival(cell2))
      return x_env.true_;
  }
  else if (are_doubles(cell1, cell2)) {
    if (fval(cell1) > fval(cell2))
      return x_env.true_;
  }
  else if (are_strs(cell1, cell2)) {
    if (strcmp(sval(cell1), sval(cell2)) > 0)
      return x_env.true_;
  }
  return x_env.nil;
}

x_any x_lt(x_any cell1, x_any cell2) {
  if (are_ints(cell1, cell2)) {
    if (ival(cell1) < ival(cell2))
      return x_env.true_;
  }
  else if (are_doubles(cell1, cell2)) {
    if (fval(cell1) < fval(cell2))
      return x_env.true_;
  }
  else if (are_strs(cell1, cell2)) {
    if (strcmp(sval(cell1), sval(cell2)) < 0)
      return x_env.true_;
  }
  return x_env.nil;
}
