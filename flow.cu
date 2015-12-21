#include "lisp.h"

x_any x_if(x_any clauses) {
  if (clauses == x_env.nil)
    return x_env.nil;
  if (x_eval(car(clauses)) != x_env.nil)
    return x_eval(cadr(clauses));
  else if (cddr(clauses) != x_env.nil)
    return x_eval(caddr(clauses));
  return x_env.nil;
}

x_any x_while(x_any cond, x_any clause) {
  x_any cell;
  cell = x_env.nil;
  if (cond == x_env.nil)
    return x_env.nil;
  while (x_eval(cond) != x_env.nil)
    cell = x_eval(clause);
  return cell;
}
 
x_any x_for(x_any sym, x_any set, x_any clause) {
  return x_env.nil;
}
