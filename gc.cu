#include "lisp.h"

void mark(x_any cell) {
  if (cdr(cell) != NULL)
    if (!testmark(cdr(cell)))
      mark(cdr(cell));
  if (car(cell) != NULL)
    if (!testmark(car(cell)))
      mark(car(cell));
}

x_any x_gc() {
  x_heap* h;
  x_any cell;
  int64_t freed = 0;

  h = x_heaps;
  do {
    cell = h->cells + X_HEAP_BLOCK_SIZE - 1;
    do
      clearmark(cell);
    while (--cell >= h->cells);
  } while (h = h->next);
  
  mark(x_symbol);
  mark(x_nil);
  mark(x_true);
  mark(x_dot);
  mark(x_lparen);
  mark(x_rparen);
  mark(x_lbrack);
  mark(x_rbrack);
  mark(x_eof);
  mark(x_builtin);
  mark(x_token);
  mark(x_user);
  mark(x_pair);
  mark(x_xector);
  mark(x_int);
  mark(x_fn0);
  mark(x_fn1);
  mark(x_fn2);
  mark(x_fn3);

  for (int i = 0; i < X_HASH_TABLE_SIZE; i++)
    mark(x_frames->names[i]);

  h = x_heaps;
  do {
    cell = h->cells + X_HEAP_BLOCK_SIZE - 1;
    do
      if (!testmark(cell)) {
        free_cell(h, cell);
        freed++;
      }
    while (--cell >= h->cells);
  } while (h = h->next);
  cell = new_cell(NULL, x_int);
  set_val(cell, freed);
  return cell;
}