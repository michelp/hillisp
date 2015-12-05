#include "lisp.h"

x_any x_if(x_any clauses) {
  if (clauses == x_nil)
    return x_nil;
  if (x_eval(car(clauses)) != x_nil)
    return x_eval(cadr(clauses));
  else if (cddr(clauses) != x_nil)
    return x_eval(caddr(clauses));
  return x_nil;
}

x_any x_while(x_any cond, x_any clause) {
  x_any cell;
  cell = x_nil;
  if (cond == x_nil)
    return x_nil;
  while (x_eval(cond) != x_nil)
    cell = x_eval(clause);
  return cell;
}

