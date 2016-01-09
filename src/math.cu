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

x_any _x_add(x_any a, x_any b, bool assign) {
  x_any c;
  if (are_ixectors(a, b)) {
    assert_xectors_align(a, b);
    if (assign)
      c = a;
    else
      c = new_ixector(xector_size(a));

    SYNCS(x_env.stream);
    xd_add<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_dxectors(a, b)) {
    assert_xectors_align(a, b);
    if (assign)
      c = a;
    else
      c = new_dxector(xector_size(a));
    SYNCS(x_env.stream);
    xd_add<double><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<double>(a), cars<double>(b), cars<double>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_ints(a, b)) {
    if (assign) {
      set_val(a, ival(a) + ival(b));
      return a;
    }
    else {
      return new_int(ival(a) + ival(b));
    }
  }
  else if (are_doubles(a, b)) {
    if (assign) {
      dval(a) = dval(a) + dval(b);
      return a;
    }
    else {
      return new_double(dval(a) + dval(b));
    }
  }
  else if (are_dcomplex(a, b)) {
    if (assign) {
      cval(a) = cuCadd(cval(a), cval(b));
      return a;
    }
    else {
      return new_dcomplex(cuCadd(cval(a), cval(b)));
    }
  }
  assert(0);
  return x_env.nil;
}

x_any x_add(x_any a, x_any b) {
  return _x_add(a, b, false);
}

x_any x_addass(x_any a, x_any b) {
  return _x_add(a, b, true);
}

x_any _x_sub(x_any a, x_any b, bool assign) {
  x_any c;
  if (are_ixectors(a, b)) {
    assert_xectors_align(a, b);
    if (assign)
      c = a;
    else
      c = new_ixector(xector_size(a));

    SYNCS(x_env.stream);
    xd_sub<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_dxectors(a, b)) {
    assert_xectors_align(a, b);
    if (assign)
      c = a;
    else
      c = new_dxector(xector_size(a));
    SYNCS(x_env.stream);
    xd_sub<double><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<double>(a), cars<double>(b), cars<double>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_ints(a, b)) {
    if (assign) {
      set_val(a, ival(a) - ival(b));
      return a;
    }
    else {
      return new_int(ival(a) - ival(b));
    }
  }
  else if (are_doubles(a, b)) {
    if (assign) {
      dval(a) = dval(a) - dval(b);
      return a;
    }
    else {
      return new_double(dval(a) - dval(b));
    }
  }
  else if (are_dcomplex(a, b)) {
    if (assign) {
      cval(a) = cuCsub(cval(a), cval(b));
      return a;
    }
    else {
      return new_dcomplex(cuCsub(cval(a), cval(b)));
    }
  }
  assert(0);
  return x_env.nil;
}


x_any x_sub(x_any a, x_any b) {
  return _x_sub(a, b, false);
}

x_any x_subass(x_any a, x_any b) {
  return _x_sub(a, b, true);
}

x_any _x_mul(x_any a, x_any b, bool assign) {
  x_any c;
  if (are_ixectors(a, b)) {
    assert_xectors_align(a, b);
    if (assign)
      c = a;
    else
      c = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_mul<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_dxectors(a, b)) {
    assert_xectors_align(a, b);
    if (assign)
      c = a;
    else
      c = new_dxector(xector_size(a));
    SYNCS(x_env.stream);
    xd_mul<double><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<double>(a), cars<double>(b), cars<double>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_ints(a, b)) {
    if (assign) {
      set_val(a, ival(a) * ival(b));
      return a;
    } else {
      return new_int(ival(a) * ival(b));
    }
  }
  else if (are_doubles(a, b)) {
    if (assign) {
      dval(a) = dval(a) * dval(b);
      return a;
    } else {
      return new_double(dval(a) * dval(b));
    }
  }
  else if (are_dcomplex(a, b)) {
    if (assign) {
      cval(a) = cuCmul(cval(a), cval(b));
    } else {
      return new_dcomplex(cuCmul(cval(a), cval(b)));
    }
  }
  assert(0);
  return x_env.nil;
}

x_any x_mul(x_any a, x_any b) {
  return _x_mul(a, b, false);
}

x_any x_mulass(x_any a, x_any b) {
  return _x_mul(a, b, true);
}

x_any _x_div(x_any a, x_any b, bool assign) {
  x_any c;
  if (are_ixectors(a, b)) {
    assert_xectors_align(a, b);
    if (assign)
      c = a;
    else
      c = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_div<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_dxectors(a, b)) {
    assert_xectors_align(a, b);
    if (assign)
      c = a;
    else
      c = new_dxector(xector_size(a));
    SYNCS(x_env.stream);
    xd_div<double><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<double>(a), cars<double>(b), cars<double>(c), xector_size(a));
    CHECK;
    return c;
  }
  else if (are_ints(a, b)) {
    if (assign) {
      set_val(a, ival(a) / ival(b));
      return a;
    } else {
      return new_int(ival(a) / ival(b));
    }
  }
  else if (are_doubles(a, b)) {
    if (assign) {
      dval(a) = dval(a) / dval(b);
      return a;
    } else {
      return new_double(dval(a) / dval(b));
    }
  }
  else if (are_dcomplex(a, b)) {
    if (assign) {
      cval(a) = cuCdiv(cval(a), cval(b));
      return a;
    } else {
      return new_dcomplex(cuCdiv(cval(a), cval(b)));
    }
  }
  assert(0);
  return x_env.nil;
}

x_any x_div(x_any a, x_any b) {
  return _x_div(a, b, false);
}

x_any x_divass(x_any a, x_any b) {
  return _x_div(a, b, true);
}

x_any _x_fma(x_any a, x_any b, x_any c, bool assign) {
  x_any d;
  if (are_ixectors(a, b)) {
    assert_xectors_align(a, b);
    assert_xectors_align(a, c);
    if (assign)
      d = a;
    else
      d = new_ixector(xector_size(a));
    SYNCS(x_env.stream);
    xd_fma<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(a), cars<int64_t>(b), cars<int64_t>(c), cars<int64_t>(d), xector_size(a));
    CHECK;
    return d;
  }
  else if (are_ints(a, b)) {
    if (assign) {
      set_val(a, ival(a) * ival(b) + ival(c));
      return a;
    } else {
      return new_int(ival(a) * ival(b) + ival(c));
    }
  }
  else if (are_doubles(a, b)) {
    if (assign) {
      dval(a) = dval(a) * dval(b) + dval(c);
      return a;
    } else {
      return new_double(dval(a) * dval(b) + dval(c));
    }
  }
  else if (are_dcomplex(a, b)) {
    if (assign) {
      cval(a) = cuCadd(cuCmul(cval(a), cval(b)), cval(c));
      return a;
    } else {
      return new_dcomplex(cuCadd(cuCmul(cval(a), cval(b)), cval(c)));
    }
  }
  assert(0);
  return x_env.nil;
}

x_any x_fma(x_any a, x_any b, x_any c) {
  return _x_fma(a, b, c, false);
}

x_any x_fmaass(x_any a, x_any b, x_any c) {
  return _x_fma(a, b, c, true);
}

x_any x_complex(x_any real, x_any imag) {
  return new_dcomplex(make_cuDoubleComplex(dval(real), dval(imag)));
}
