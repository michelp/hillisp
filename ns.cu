#include "lisp.h"

int hash(const char *name) {
  int value = 0;
  while (*name != '\0')
    value = (value * X_HASH_MULTIPLIER + *name++) % X_HASH_TABLE_SIZE;
  return value;
}

x_any _lookup(const char *name, x_any binding) {
  if (binding == x_env.nil)
    return NULL;
  else if (strcmp(sval(binding), name) == 0)
    return binding;
  else
    return _lookup(name, cdr(binding));
}

x_any lookup(const char* name) {
  return _lookup(name, x_env.frames->names[hash(name)]);
}

void bind(const char* name, x_any cell1, x_frame* frame) {
  int hash_val;
  x_any cell, cell2;
  hash_val = hash(name);
  cell2 = frame->names[hash_val];

  cell = new_cell(name, x_env.binding);
  set_car(cell, cell1);
  set_cdr(cell, cell2);
  frame->names[hash_val] = cell;
}

void rebind(const char* name, x_any value) {
  x_any binding;
  binding = lookup(name);
  assert(binding != NULL);
}

x_any intern(const char *name) {
  x_any cell;
  cell = lookup(name);
  if (cell != NULL)
    return car(cell);

  cell = new_cell(name, x_env.symbol);
  bind(name, cell, x_env.frames);
  return cell;
}
