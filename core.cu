#include <sys/time.h>
#include "lisp.h"

template<typename T>
__global__ void
 xd_fill(T* __restrict__ cars, const T val, const size_t size) {
  int i = TID;
  while (i < size) {
    cars[i] = val;
    i += STRIDE;
  }
}

template<typename T>
__global__ void
 xd_all(const T* __restrict__ cell, int* __restrict__ result, const size_t size) {
  if (*result == size)
    if (TID < size)
      if (!cell[TID])
        atomicSub(result, 1);
  __syncthreads();
}

template<typename T>
__global__ void
 xd_any(const T* __restrict__ cell, int* __restrict__ result, const size_t size) {
  if (*result == 0)
    if (TID < size)
      if (cell[TID])
        atomicAdd(result, 1);
  __syncthreads();
}

x_any x_is(x_any cell1, x_any cell2) {
  if (cell1 == cell2)
    return x_env.true_;
  return x_env.nil;
}

x_any x_isinstance(x_any cell1, x_any cell2) {
  do {
    cell1 = type(cell1);
    if (cell1 == cell2)
      return x_env.true_;
  } while(cell1 != x_env.symbol);
  return x_env.nil;
}

x_any x_type(x_any cell) {
  return cell->type;
}

x_any x_assert(x_any cell) {
  assert(cell != x_env.nil);
  return cell;
}

x_any x_car(x_any cell) {
  return car(cell);
}

x_any x_cdr(x_any cell) {
  return cdr(cell);
}

x_any inline x_cons(x_any cell1, x_any cell2) {
  x_any cell;
  cell = new_cell(NULL, x_env.pair);
  set_car(cell, cell1);
  set_cdr(cell, cell2);
  return cell;
}

x_any x_list(x_any args) {
  return args;
}

x_any x_apply(x_any cell, x_any args) {
  x_any expr, result;
  if (is_special(cell))
    return ((x_fn1)val(cell))(args);

  args = eval_list(args);
  if (is_builtin(cell)) {
    if (is_fn1(cell))
      return ((x_fn1)val(cell))(car(args));
    else if (is_fn2(cell))
      return ((x_fn2)val(cell))(car(args), cadr(args));
    else if (is_fn3(cell))
      return ((x_fn3)val(cell))(car(args), cadr(args), caddr(args));
    else if (is_fn0(cell))
      return ((x_fn0)val(cell))();
    else if (is_fnv(cell))
      return ((x_fnv)val(cell))(args);
    else
      assert(0);
  }
  else if (is_user(cell)) {
    expr = car(cell);
    assert(length(args) == length(expr));
    push_frame();

    do {
      local(sval(car(expr)), car(args));
      expr = cdr(expr);
      args = cdr(args);
    } while(expr != x_env.nil);

    expr = cdr(cell);
    do {
      result = x_eval(car(expr));
      expr = cdr(expr);
    } while (expr != x_env.nil);
    pop_frame();
    return result;
  }
  else if (is_symbol(cell) || is_int(cell))
    return x_cons(cell, args);
  else if (is_pair(cell))
    return x_cons(x_eval(cell), args);
  else
    assert(0);
  return x_env.nil;
}

x_any x_quote(x_any args) {
  return car(args);
}

x_any eval_symbol(x_any sym) {
  char* name;
  x_any cell;
  name = sval(sym);
  assert(name != NULL);
  if (isdigit(name[0]) || (name[0] == '-' && isdigit(name[1]))) {
    if (strchr(name, '.') == NULL)
      return new_int(strtoll(name, NULL, 0));
    else
      return new_double(strtod(name, NULL));
  }
  cell = lookup(name, -1);
  if (cell == NULL)
    return sym;
  return car(cell);
}

x_any eval_list(x_any cell) {
  if (cell == x_env.nil)
    return x_env.nil;
  if (is_symbol(cell))
    return eval_symbol(cell);
  else if (is_atom(cell))
    return cell;
  else
    return x_cons(x_eval(car(cell)), eval_list(cdr(cell)));
}

x_any x_eval(x_any cell) {
  x_any temp;
  if (is_symbol(cell))
      return eval_symbol(cell);
  else if (is_atom(cell))
    return cell;
  else if (is_pair(cell)) {
    temp = x_eval(car(cell));
    if (is_func(temp))
      return x_apply(temp, cdr(cell));
    else
      return x_cons(temp, eval_list(cdr(cell)));
  }
  assert(0);
  return x_env.nil;
}

x_any x_not(x_any cell) {
  if (cell == x_env.nil)
    return x_env.true_;
  return x_env.nil;
}

x_any x_and(x_any cell1, x_any cell2) {
  if (cell1 != x_env.nil && cell2 != x_env.nil)
    return x_env.true_;
  return x_env.nil;
}

x_any x_or(x_any cell1, x_any cell2) {
  if (cell1 != x_env.nil || cell2 != x_env.nil)
    return x_env.true_;
  return x_env.nil;
}

x_any x_fill(x_any value, x_any size) {
  x_any cell;
  if (!is_int(size))
    assert(0);
  if (is_int(value)) {
    cell = new_ixector(ival(size));
    xd_fill<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<int64_t>(cell), ival(value), xector_size(cell));
  }
  else if (is_double(value)) {
    cell = new_dxector(ival(size));
    xd_fill<double><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
      (cars<double>(cell), fval(value), xector_size(cell));
  }
  CHECK;
  return cell;
}

x_any x_all(x_any cell) {
  int* result;
  if (!is_xector(cell))
    assert(0);
  SYNCS(x_env.stream);
  cudaMallocManaged(&result, sizeof(int));
  assert(result != NULL);
  *result = xector_size(cell);
  xd_all<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
    (cars<int64_t>(cell), result, xector_size(cell));
  SYNCS(x_env.stream);
  CHECK;
  if (*result != xector_size(cell))
    return x_env.nil;
  return x_env.true_;
}

x_any x_any_(x_any cell) {
  int* result;
  if (!is_xector(cell))
    assert(0);
  SYNCS(x_env.stream);
  cudaMallocManaged(&result, sizeof(int));
  assert(result != NULL);
  *result = 0;
  xd_any<int64_t><<<BLOCKS, THREADSPERBLOCK, 0, x_env.stream>>>
    (cars<int64_t>(cell), result, xector_size(cell));
  SYNCS(x_env.stream);
  CHECK;
  if (*result > 0)
    return x_env.true_;
  return x_env.nil;
}

x_any x_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return new_int((tv.tv_sec * 1000) + tv.tv_usec);
}

x_any x_set(x_any args) {
  return bind(sval(car(args)), x_eval(cadr(args)));
}

int64_t inline length(x_any cell) {
  int64_t length = 0;
  if (is_xector(cell))
    return xector_size(cell);
  else if (cdr(cell) == NULL)
    return 0;
  else
    do {
      length += 1;
      cell = cdr(cell);
    } while(cell != x_env.nil);
  return length;
}

x_any x_len(x_any cell) {
  return new_int(length(cell));
}
