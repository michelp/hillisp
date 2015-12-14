#include "lisp.h"

void mark(x_any cell) {
  while (cell != NULL) {
    if (!((int64_t)(cell->cdr)) & 1)
      return;
    *(int64_t*)&cdr(cell) &= ~1;
    if (!is_func(cell))
        mark(car(cell));
    cell = cdr(cell);
  }
}

x_any x_gc() {
  x_heap* heap;
  x_any cell;
  int64_t freed = 0;

   heap = x_heaps;
   do {
      cell = heap->cells + X_HEAP_BLOCK_SIZE-1;
      do
         *(int64_t*)&cdr(cell) |= 1;
      while (--cell >= heap->cells);
   } while (heap = heap->next);

  mark(x_dot);
  mark(x_lparen);
  mark(x_rparen);
  mark(x_lbrack);
  mark(x_rbrack);
  mark(x_eof);

  for (int i = 0; i < X_HASH_TABLE_SIZE; i++)
    mark(x_frames->names[i]);

   heap = x_heaps;
   SYNCS(stream);
   do {
     cell = heap->cells + X_HEAP_BLOCK_SIZE-1;
     do
       if ((int64_t)(cell->cdr) & 1) {
         if (is_xector(cell)) {
           if (xval(cell)->cars != NULL) {
             cudaFree(xval(cell)->cars);
             CHECK;
             xval(cell)->cars = NULL;
           }
           free(xval(cell));
         }
         cell->type = NULL;
         cell->value = NULL;
         cell->car = NULL;
         free_cell(heap, cell);
         freed++;
       }
     while (--cell >= heap->cells);
   } while (heap = heap->next);
   cell = new_cell(NULL, x_int);
   set_val(cell, freed);
   return cell;
}