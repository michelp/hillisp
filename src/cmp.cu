#include "lisp.h"

template<typename T>
__global__ void
 xd_eq(const T* __restrict__ a, const T* __restrict__ b, int64_t* __restrict__ c, const size_t size) {
  int i = TID;
  while (i < size) {
    c[i] = (int64_t)(a[i] == b[i]);
    i += STRIDE;
  }
}

__global__ void
 xd_dceq(const cuDoubleComplex* __restrict__ a, const cuDoubleComplex* __restrict__ b, int64_t* __restrict__ c, const size_t size) {
  int i = TID;
  while (i < size) {
    c[i] = (int64_t)(a[i].x == b[i].x && a[i].y == b[i].y);
    i += STRIDE;
  }
}

template<typename T>
__global__ void
 xd_gt(const T* __restrict__ a, const T* __restrict__ b, int64_t* __restrict__ c, const size_t size) {
  int i = TID;
  while (i < size) {
    c[i] = (int64_t)(a[i] > b[i]);
    i += STRIDE;
  }
}

template<typename T>
__global__ void
 xd_lt(const T* __restrict__ a, const T* __restrict__ b, int64_t* __restrict__ c, const size_t size) {
  int i = TID;
  while (i < size) {
    c[i] = (int64_t)(a[i] < b[i]);
    i += STRIDE;
  }
}

template<typename T>
__global__ void
 xd_gte(const T* __restrict__ a, const T* __restrict__ b, int64_t* __restrict__ c, const size_t size) {
  int i = TID;
  while (i < size) {
    c[i] = (int64_t)(a[i] >= b[i]);
    i += STRIDE;
  }
}

template<typename T>
__global__ void
 xd_lte(const T* __restrict__ a, const T* __restrict__ b, int64_t* __restrict__ c, const size_t size) {
  int i = TID;
  while (i < size) {
    c[i] = (int64_t)(a[i] <= b[i]);
    i += STRIDE;
  }
}

x_any x_eq(x_any a, x_any b) {
  x_any c;
  if (a == b)
    return x_env.true_;
  else if (are_ixectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_eq<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_dxectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_eq<double><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<double>(a), cars<double>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_dcxectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_dceq<<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<cuDoubleComplex>(a), cars<cuDoubleComplex>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_symbols(a, b)) {
    if (strcmp(sval(a), sval(b)) == 0)
      return x_env.true_;
  }
  else if (are_ints(a, b)) {
    if (ival(a) == ival(b))
      return x_env.true_;
  }
  else if (are_doubles(a, b)) {
    if (dval(a) == dval(b))
      return x_env.true_;
  }
  else if (are_dcomplex(a, b)) {
    if (crval(a) == crval(b) && cival(a) == cival(b))
      return x_env.true_;
  }
  else if (are_strs(a, b)) {
    if (strcmp(sval(a), sval(b)) == 0)
      return x_env.true_;
  }
  else if (are_pairs(a, b)) {
    do {
      if (x_eq(car(a), car(b)) == x_env.nil)
        return x_env.nil;
      a = cdr(a);
      b = cdr(b);
    } while (are_pairs(a, b));
    if (x_eq(a, b) != x_env.nil)
      return x_env.true_;
  }
  return x_env.nil;
}

x_any x_neq(x_any a, x_any b) {
  if (x_eq(a, b) == x_env.true_)
    return x_env.nil;
  return x_env.true_;
}

x_any x_gt(x_any a, x_any b) {
  x_any c;
  if (are_ixectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_gt<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_dxectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_gt<double><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<double>(a), cars<double>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_ints(a, b)) {
    if (ival(a) > ival(b))
      return x_env.true_;
  }
  else if (are_doubles(a, b)) {
    if (dval(a) > dval(b))
      return x_env.true_;
  }
  else if (are_strs(a, b)) {
    if (strcmp(sval(a), sval(b)) > 0)
      return x_env.true_;
  }
  return x_env.nil;
}

x_any x_lt(x_any a, x_any b) {
  x_any c;
  if (are_ixectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_lt<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_dxectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_lt<double><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<double>(a), cars<double>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_ints(a, b)) {
    if (ival(a) < ival(b))
      return x_env.true_;
  }
  else if (are_doubles(a, b)) {
    if (dval(a) < dval(b))
      return x_env.true_;
  }
  else if (are_strs(a, b)) {
    if (strcmp(sval(a), sval(b)) < 0)
      return x_env.true_;
  }
  return x_env.nil;
}

x_any x_gte(x_any a, x_any b) {
  x_any c;
  if (are_ixectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_gte<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_dxectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_gte<double><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<double>(a), cars<double>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_ints(a, b)) {
    if (ival(a) > ival(b))
      return x_env.true_;
  }
  else if (are_doubles(a, b)) {
    if (dval(a) > dval(b))
      return x_env.true_;
  }
  else if (are_strs(a, b)) {
    if (strcmp(sval(a), sval(b)) > 0)
      return x_env.true_;
  }
  return x_env.nil;
}

x_any x_lte(x_any a, x_any b) {
  x_any c;
  if (are_ixectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_lte<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_dxectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_lte<double><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<double>(a), cars<double>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_ints(a, b)) {
    if (ival(a) < ival(b))
      return x_env.true_;
  }
  else if (are_doubles(a, b)) {
    if (dval(a) < dval(b))
      return x_env.true_;
  }
  else if (are_strs(a, b)) {
    if (strcmp(sval(a), sval(b)) < 0)
      return x_env.true_;
  }
  return x_env.nil;
}
