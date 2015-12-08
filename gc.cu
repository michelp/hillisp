#include "lisp.h"


void mark(x_any cell) {
  while (cell != NULL) {
    if (!cdr<uint64_t>(cell) & 1)
      return;
    *(uint64_t*)&(cell->cdr) &= ~1;
    mark(car<x_any>(cell));
    cell = cdr<x_any>(cell);
  }
  if ((car<uint64_t>(cell)) & 1) {

    mark(car<x_any>(cell));
    cell = cdr<x_any>(cell);
    *(uint64_t*)&(cell->car) &= ~1;
    while (cell != NULL) {
      if (!cdr<uint64_t>(cell) & 1)
        return;
      *(uint64_t*)&(cell->cdr) &= ~1;
      mark(cdr<x_any>(cell));
      cell = car<x_any>(cell);
    }
  }
}

static void gc() {
  x_any cell;
  x_heap *h;

   h = x_heaps;
   do {
     cell = h->cells + X_HEAP_BLOCK_SIZE - 1;
     do
       *(uint64_t*)&(cell->cdr) |= 1;
     while (--cell >= h->cells);
   } while (h = h->next);

   mark(x_nil);
   mark(x_symbol);
   mark(x_garbage);
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
}
