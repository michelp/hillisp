#include "lisp.h"

void mark(x_any cell) {
  while (cell != NULL) {
    if (!((int64_t)(cell->cdr)) & 1)
      return;
    *(int64_t*)&cdr(cell) &= ~1;
    mark(car(cell));
    cell = cdr(cell);
  }
}

x_any x_gc() {
  x_cell_pool* cell_pool;
  x_any cell;
  int64_t freed = 0;

   cell_pool = x_env.cell_pools;
   do {
      cell = cell_pool->cells + X_YOUNG_CELL_POOL_SIZE-1;
      do
         *(int64_t*)&cdr(cell) |= 1;
      while (--cell >= cell_pool->cells);
   } while (cell_pool = cell_pool->next);

  mark(x_env.dot);
  mark(x_env.lparen);
  mark(x_env.rparen);
  mark(x_env.lbrack);
  mark(x_env.rbrack);
  mark(x_env.eof);

  for (int i = 0; i < X_HASH_TABLE_SIZE; i++)
    mark(x_env.frames->names[i]);

   cell_pool = x_env.cell_pools;
   SYNCS(x_env.stream);
   do {
     cell = cell_pool->cells + X_YOUNG_CELL_POOL_SIZE-1;
     do
       if ((int64_t)(cell->cdr) & 1) {
         if (is_xector(cell)) {
           if (xval(cell) != NULL) {
             cudaFree(xval(cell));
             //printf("xfree\n");
             CHECK;
           }
         }
         cell->type = NULL;
         cell->value = NULL;
         cell->car = NULL;
         free_cell(cell_pool, cell);
         freed++;
       }
     while (--cell >= cell_pool->cells);
   } while (cell_pool = cell_pool->next);
   cell = new_cell(NULL, x_env.int_);
   set_val(cell, freed);
   return cell;
}