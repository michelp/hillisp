#include <sys/time.h>
#include "lisp.h"

template<typename T>
__global__ void
 xd_fill(T *cars, T val, size_t size) {
  if (TID < size)
    cars[TID] = val;
}

template<typename T>
__global__ void
 xd_all(T* cell1, int *result, size_t size) {
  if (*result == size)
    if (TID < size)
      if (!cell1[TID])
        atomicSub(result, 1);
  __syncthreads();
}

template<typename T>
__global__ void
 xd_any(T* cell1, int *result, size_t size) {
  if (*result == 0)
    if (TID < size)
      if (cell1[TID])
        atomicAdd(result, 1);
  __syncthreads();
}

x_any x_is(x_any cell1, x_any cell2) {
  if (cell1 == cell2)
    return x_true;
  return x_nil;
}

x_any x_isinstance(x_any cell1, x_any cell2) {
  do {
    cell1 = type(cell1);
    if (cell1 == cell2)
      return x_true;
  } while(cell1 != x_symbol);
  return x_nil;
}

x_any x_type(x_any cell) {
  return cell->type;
}

x_any x_assert(x_any cell) {
  assert(cell != x_nil);
  return cell;
}

x_any x_car(x_any cell) {
  if (!is_pair(cell))
    assert(0);
  return car(cell);
}

x_any x_cdr(x_any cell) {
  if (!is_pair(cell))
    assert(0);
  return cdr(cell);
}

x_any x_cons(x_any cell1, x_any cell2) {
  x_any cell;
  cell = new_cell(NULL, x_pair);
  set_car(cell, cell1);
  set_cdr(cell, cell2);
  return cell;
}

x_any x_apply(x_any cell, x_any args) {
  if (is_symbol(cell) && !(is_func(cell)))
    return x_cons(cell, args);
  if (is_pair(cell))
    return x_cons(x_eval(cell), args);
  if (is_builtin(cell)) {
#ifdef DEBUG
    printf("%*s" "%s\n", debugLevel, " ", name(cell));
#endif
    if (is_fn0(cell))
      return ((x_fn0_t)cdr(cell))();
    else if (is_fn1(cell))
      return ((x_fn1_t)cdr(cell))(car(args));
    else if (is_fn2(cell))
      return ((x_fn2_t)cdr(cell))(car(args), cadr(args));
    else if (is_fn3(cell))
      return ((x_fn3_t)cdr(cell))(car(args), cadr(args), caddr(args));
    else
      assert(0);
  }
  else if (is_user(cell))
    return x_apply((x_any)cdr(cell), args);
  else
    assert(0);
  return x_nil;
}

x_any x_quote(x_any cell) {
  return cell;
}

x_any x_eval(x_any cell) {
  x_any temp;
  if (is_atom(cell))
    return cell;
  else if (is_pair(cell) && (is_func(car(cell)))) {
#ifdef DEBUG
    debugLevel += 2;
#endif
    temp = x_apply(car(cell), list_eval(cdr(cell)));
#ifdef DEBUG
    debugLevel -= 2;
#endif
    return temp;
  }
  else {
    temp = x_eval(car(cell));
    return x_cons(temp, list_eval(cdr(cell)));
  }
}

x_any x_not(x_any cell1) {
  if (cell1 == x_true)
    return x_nil;
  return x_true;
}

x_any x_and(x_any cell1, x_any cell2) {
  if (cell1 == x_true && cell2 == x_true)
    return x_true;
  return x_nil;
}

x_any x_or(x_any cell1, x_any cell2) {
  if (cell1 == x_true || cell2 == x_true)
    return x_true;
  return x_nil;
}

x_any x_fill(x_any val, x_any size) {
  x_any cell;
  if (!is_int(size))
    assert(0);
  cell = new_xector(NULL, int64_car(size));
  xd_fill<int64_t><<<GRIDBLOCKS(xector_size(cell)), THREADSPERBLOCK, 0, stream>>>
    (int64_cars(cell), int64_car(val), xector_size(cell));
  CHECK;
  return cell;
}

x_any x_all(x_any cell) {
  int* result;
  if (!is_xector(cell))
    assert(0);
  SYNCS(stream);
  cudaMallocManaged(&result, sizeof(int));
  assert(result != NULL);
  *result = xector_size(cell);
  xd_all<int64_t><<<GRIDBLOCKS(xector_size(cell)), THREADSPERBLOCK, 0, stream>>>
    (int64_cars(cell), result, xector_size(cell));
  SYNCS(stream);
  CHECK;
  if (*result != xector_size(cell))
    return x_nil;
  return x_true;
}

x_any x_any_(x_any cell) {
  int* result;
  if (!is_xector(cell))
    assert(0);
  SYNCS(stream);
  cudaMallocManaged(&result, sizeof(int));
  assert(result != NULL);
  *result = 0;
  xd_any<int64_t><<<GRIDBLOCKS(xector_size(cell)), THREADSPERBLOCK, 0, stream>>>
    (int64_cars(cell), result, xector_size(cell));
  SYNCS(stream);
  CHECK;
  if (*result > 0)
    return x_true;
  return x_nil;
}

x_any x_time() {
  x_any cell;
  cell = new_cell(NULL, x_int);
  struct timeval tv;
  gettimeofday(&tv,NULL);
  set_car(cell, tv.tv_sec*(uint64_t)1000000+tv.tv_usec);
  return cell;
}

