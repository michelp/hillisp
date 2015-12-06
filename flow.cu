#include "lisp.h"

x_any x_if(x_any clauses) {
  if (clauses == x_nil)
    return x_nil;
  if (x_eval(carr<x_any>(clauses)) != x_nil)
    return x_eval(cadr<x_any>(clauses));
  else if (cddr<x_any>(clauses) != x_nil)
    return x_eval(caddr<x_any>(clauses));
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

x_any x_for(x_any sym, x_any set, x_any clause) {
  return x_nil;
}

