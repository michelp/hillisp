#include "lisp.h"

x_any x_if(x_any args) {
  x_any cond, body, result;
  cond = car(args);
  args = cdr(args);
  result = x_env.nil;

  if (x_eval(cond) != x_env.nil) {
    body = car(args);
    do {
      result = x_eval(car(body));
      body = cdr(body);
    } while (body != x_env.nil);
  }
  else if (cdr(args) != x_env.nil) {
    body = cadr(args);
    do {
      result = x_eval(car(body));
      body = cdr(body);
    } while (body != x_env.nil);
  }
  return result;
}

x_any x_while(x_any args) {
  x_any cond, body, collection, result;
  cond = x_eval(car(args));
  body = cadr(args);
  collection = x_env.nil;
  result = x_env.nil;

  push_frame();
  collection = local("@@", collection);

  while (cond != x_env.nil) {
    do {
      result = x_eval(car(body));
      body = cdr(body);
    } while (body != x_env.nil);
  }
  pop_frame();
  return result;
}
 
x_any x_do(x_any args) {
  x_any count, body, collection, result;
  count = x_eval(car(args));
  assert(is_int(count));
  collection = x_env.nil;
  result = x_env.nil;

  push_frame();
  collection = local("@@", collection);

  for (int i = 0; i < ival(count); i++) {
    body = cdr(args);
    do {
      result = x_eval(car(body));
      body = cdr(body);
    } while (body != x_env.nil);
  }
  pop_frame();
  return result;
}

x_any x_for(x_any args) {
  char* index;
  x_any sym, start, end, body, collection, result;
  
  result = x_env.nil;
  collection = x_env.nil;
  sym = car(args);
  index = sval(sym);
  start = x_eval(cadr(args));
  if (start == x_env.nil)
    return start;

  if (is_int(start)) {
    end = x_eval(caddr(args));
    push_frame();
    sym = local(index, start);
    args = local("@", args);
    start = local("@<", start);
    end = local("@>", end);
    collection = local("@@", collection);

    while (ival(sym) < ival(end)) {
      body = cdddr(args);
      do {
        result = x_eval(car(body));
        body = cdr(body);
      } while (body != x_env.nil);
      sym = local(index, new_int(ival(sym) + 1));
    }
  } else if (is_pair(start)) {
    push_frame();
    sym = local(index, car(start));
    sym = local("@", args);
    collection = local("@@", collection);
    do {
      body = cddr(args);
      do {
        result = x_eval(car(body));
        body = cdr(body);
      } while (body != x_env.nil);
      start = cdr(start);
      sym = local(index, car(start));
    } while (start != x_env.nil);
  } 
  else
    assert(0);
  pop_frame();
  return result;
}

x_any x_collect(x_any cell) {
  x_any collection;
  collection = lookup("@@", 0);
  if (collection == NULL)
    return x_env.nil;
  return bind("@@", x_cons(x_eval(cell), car(collection)));
}
