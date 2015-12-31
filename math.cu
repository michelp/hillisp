#include "lisp.h"

template<typename T>
__global__ void 
xd_add(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, const size_t size) {
  for (int i = TID; i < size; i += STRIDE)
    c[i] = a[i] + b[i];
}

template<typename T>
__global__ void
xd_sub(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, const size_t size) {
  for (int i = TID; i < size; i += STRIDE)
    c[i] = a[i] - b[i];
}

template<typename T>
__global__ void
xd_mul(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, const size_t size) {
  for (int i = TID; i < size; i += STRIDE)
    c[i] = a[i] * b[i];
}

template<typename T>
__global__ void
xd_div(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, const size_t size) {
  for (int i = TID; i < size; i += STRIDE)
    c[i] = a[i] / b[i];
}

template<typename T>
__global__ void
xd_fma(const T* __restrict__ a, const T* __restrict__ b, const T* __restrict__ c, T* __restrict__ d, const size_t size) {
  for (int i = TID; i < size; i += STRIDE)
    d[i] = a[i] * b[i] + c[i];
}

x_any x_add(x_any a, x_any b) {
  x_any c;
  if (are_xectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_xector<int64_t>(NULL, xector_size(a));
    SYNCS(x_env.stream);
    xd_add<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_ints(a, b))
    return new_int(ival(a) + ival(b));
  else if (are_floats(a, b))
    return new_float(fval(a) + fval(b));
  assert(0);
  return x_env.nil;
}

x_any x_sub(x_any a, x_any b) {
  x_any c;
  if (are_xectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_xector<int64_t>(NULL, xector_size(a));
    SYNCS(x_env.stream);
    xd_sub<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_ints(a, b))
    return new_int(ival(a) - ival(b));
  else if (are_floats(a, b))
    return new_float(fval(a) - fval(b));
  assert(0);
  return x_env.nil;
}

x_any x_mul(x_any a, x_any b) {
  x_any c;
  if (are_xectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_xector<int64_t>(NULL, xector_size(a));
    SYNCS(x_env.stream);
    xd_mul<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_ints(a, b))
    return new_int(ival(a) * ival(b));
 else if (are_floats(a, b))
    return new_float(fval(a) * fval(b));
   assert(0);
  return x_env.nil;
}

x_any x_div(x_any a, x_any b) {
  x_any c;
  if (are_xectors(a, b)) {
    assert_xectors_align(a, b);
    c = new_xector<int64_t>(NULL, xector_size(a));
    SYNCS(x_env.stream);
    xd_div<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_ints(a, b))
    return new_int(ival(a) / ival(b));
 else if (are_floats(a, b))
    return new_float(fval(a) / fval(b));
   assert(0);
  return x_env.nil;
}

x_any x_fma(x_any a, x_any b, x_any c) {
  x_any d;
  if (are_xectors(a, b)) {
    assert_xectors_align(a, b);
    assert_xectors_align(a, c);
    d = new_xector<int64_t>(NULL, xector_size(a));
    SYNCS(x_env.stream);
    xd_fma<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), cars<int64_t>(d), xector_size(a));
    CHECK;
    return d;
  }
  else if (are_ints(a, b))
    return new_int(ival(a) * ival(b) + ival(c));
  else if (are_floats(a, b))
    return new_float(fval(a) * fval(b) + fval(c));
  assert(0);
  return x_env.nil;
}
