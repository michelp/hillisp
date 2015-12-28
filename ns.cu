#include "lisp.h"

int inline hash(const char *name) {
  int value = 0;
  while (*name != '\0')
    value = (value * X_HASH_MULTIPLIER + *name++) % X_HASH_TABLE_SIZE;
  return value;
}

x_any lookup(const char *name) {
  x_any binding;
  int hashval;
  hashval = hash(name);

  for (int i = x_env.frame_count; i > 0; i--) {
    binding = x_env.frames[i - 1][hashval];
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

void bind(const char* name, x_any value) {
  int hash_val;
  x_any binding, bucket;
  binding = lookup(name);
  if (binding != NULL) {
    set_car(binding, value);
    return;
  }

  hash_val = hash(name);
  bucket = current_frame[hash_val];
  binding = new_cell(name, x_env.binding);
  set_car(binding, value);
  set_cdr(binding, bucket);
  current_frame[hash_val] = binding;
}

x_any intern(const char *name) {
  x_any cell;
  cell = lookup(name);
  if (cell != NULL)
    return car(cell);

  cell = new_cell(name, x_env.symbol);
  bind(name, cell);
  return cell;
}

x_any x_dir() {
  x_any binding, result;
  result = x_env.nil;

  for (int i = 0; i < x_env.frame_count; i++) {
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

x_any x_def(x_any name, x_any args, x_any body) {
  assert(is_symbol(name));
  type(name) = x_env.user;
  bind(sval(name), name);
  set_car(name, x_cons(args, body));
  return x_env.nil;
}
