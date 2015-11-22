#include "lisp.h"

x_any x_is(x_any cell1, x_any cell2) {
  if (cell1 == cell2)
    return x_true;
  return x_nil;
}

x_any x_car(x_any cell) {
  return car(cell);
}

x_any x_cdr(x_any cell) {
  return cdr(cell);
}

x_any x_cons(x_any cell1, x_any cell2) {
  x_any cell;
  cudaMallocManaged(&cell, sizeof(x_cell));
  assert(cell != NULL);
  flags(cell) = PAIR;
  car(cell) = cell1;
  cdr(cell) = cell2;
  return cell;
}


x_any x_print(x_any cell) {
  print_cell(cell, stdout);
  putchar('\n');
  return cell;
}

x_any x_apply(x_any cell, x_any args) {
  if (is_symbol(cell))
    return x_cons(cell, args);
  if (is_pair(cell))
    return x_cons(x_eval(cell), args);
  if (is_builtin(cell))
    switch (size(cell)) {
    case 0:
      return ((x_fn0)data(cell))();
    case 1:
      return ((x_fn1)data(cell))(car(args));
    case 2:
      return ((x_fn2)data(cell))(car(args), car(cdr(args)));
    case 3:
      return ((x_fn3)data(cell))(car(args), car(cdr(args)), car(cdr(cdr(args))));
    }
  else if (is_user(cell))
    return x_apply((x_any)data(cell), args);
  else
    assert(0);
  return x_nil;
}

x_any x_quote(x_any cell) {
  return cell;
}

x_any x_cond(x_any clauses) {
  if (clauses == x_nil)
    return x_nil;
  else if (x_eval(car(car(clauses))) != x_nil)
    return x_eval(car(cdr(clauses)));
  else 
    return x_cond(car(cdr(clauses)));
}

x_any x_eval(x_any cell) {
  if (is_atom(cell))
    return cell;
  else if (is_pair(cell) && (is_symbol(car(cell))))
    return x_cons(car(cell), list_eval(cdr(cell)));
  else
    return x_apply(car(cell), list_eval(cdr(cell)));
}
