#include "lisp.h"

int inline hash(const char *name) {
  int value = 0;
  while (*name != '\0')
    value = (value * X_HASH_MULTIPLIER + *name++) % X_HASH_TABLE_SIZE;
  return value;
}

x_any lookup(const char *name, int depth) {
  x_any binding;
  int hashval;
  hashval = hash(name);
  if (depth == -1)
    depth = 0;
  else
    depth = x_env.frame_count - depth;

  for (int i = x_env.frame_count; i >= depth; i--) {
    binding = get_frame_bucket(i, hashval);
    if (binding == x_env.nil)
      continue;
    do {
      if (strcmp(name, sval(binding)) == 0)
        return binding;
      binding = cdr(binding);
    } while (binding != x_env.nil);
  }
  return NULL;
}

x_any _bind(const char* name, x_any value, int depth) {
  int hash_val;
  x_any binding, bucket;
  binding = lookup(name, depth);
  if (binding != NULL) {
    set_car(binding, value);
  } else {
    hash_val = hash(name);
    bucket = current_frame_bucket(hash_val);
    binding = new_cell(name, x_env.binding);
    set_car(binding, value);
    set_cdr(binding, bucket);
    current_frame_bucket(hash_val) = binding;
  }
  return value;
}

x_any bind(const char* name, x_any value) {
  return _bind(name, value, -1);
}

x_any local(const char* name, x_any value) {
  return _bind(name, value, 0);
}

x_any intern(const char *name) {
  x_any cell;
  cell = lookup(name, -1);
  if (cell != NULL)
    return car(cell);

  cell = new_cell(name, x_env.symbol);
  bind(name, cell);
  return cell;
}

x_any x_dir() {
  x_any binding, result;
  result = x_env.nil;

  for (int i = x_env.frame_count; i <= 0; i--) {
    for (int j = 0; j < X_HASH_TABLE_SIZE; j++) {
      binding = x_env.frames[i][j];
      if (binding != x_env.nil) {
        do {
          result = x_cons(binding, result);
          binding = cdr(binding);
        } while (binding != x_env.nil);
      }
    }
  }
  return result;
}

x_any x_def(x_any args) {
  x_any name;
  name = car(args);
  assert(is_symbol(name));
  set_type(name, x_env.user);
  set_car(name, car(cdr(args)));
  set_cdr(name, cdr(cdr(args)));
  local(sval(name), name);
  return name;
}
