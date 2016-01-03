#include "lisp.h"

x_any x_if(x_any args) {
  x_any cond;
  cond = car(args);
  args = cdr(args);
  if (x_eval(cond) != x_env.nil)
    return x_eval(car(args));
  else if (cdr(args) != x_env.nil)
    return x_eval(cadr(args));
  return x_env.nil;
}

x_any x_while(x_any args) {
  x_any cond, clause, result;
  cond = x_eval(car(args));
  clause = cdr(args);
  result = x_env.nil;

  while (cond != x_env.nil) {
    result = x_eval(clause);
    cond = x_eval(cond);
  }
  return result;
}
 
x_any x_do(x_any args) {
  x_any result, count, body;

  count = x_eval(car(args));
  assert(is_int(count));

  for (int i = 0; i < ival(count); i++) {
    body = cdr(args);
    do {
      result = x_eval(car(body));
      body = cdr(body);
    } while (body != x_env.nil);
  }
  return result;
}

x_any x_for(x_any args) {
  return x_env.nil;
}