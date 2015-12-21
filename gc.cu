#include "lisp.h"

void mark(x_any cell) {
  while (cell != NULL) {
    if (!((int64_t)(cell->cdr)) & 1)
      return;
    *(int64_t*)&cdr(cell) &= ~1;
    if (is_xector(cell))
      *(int64_t*)&val(cell) &= ~1;

    mark(car(cell));
    cell = cdr(cell);
  }
}

x_any x_gc() {
  x_cell_pool* cell_pool;
  x_any cell;
  int64_t freed = 0;

   cell_pool = x_env.x_cell_pools;
   do {
      cell = cell_pool->cells + X_YOUNG_CELL_POOL_SIZE-1;
      do
         *(int64_t*)&cdr(cell) |= 1;
      while (--cell >= cell_pool->cells);
   } while (cell_pool = cell_pool->next);

  mark(x_env.x_dot);
  mark(x_env.x_lparen);
  mark(x_env.x_rparen);
  mark(x_env.x_lbrack);
  mark(x_env.x_rbrack);
  mark(x_env.x_eof);

  for (int i = 0; i < X_HASH_TABLE_SIZE; i++)
    mark(x_env.x_frames->names[i]);

   cell_pool = x_env.x_cell_pools;
   SYNCS(x_env.stream);
   do {
     cell = cell_pool->cells + X_YOUNG_CELL_POOL_SIZE-1;
     do
       if ((int64_t)(cell->cdr) & 1) {
         if (is_xector(cell)) {
           if (xval(cell)->cars != NULL) {
             cudaFree(xval(cell)->cars);
             printf("xfree\n");
             CHECK;
             xval(cell)->cars = NULL;
           }
           free(xval(cell));
         }
         cell->type = NULL;
         cell->value = NULL;
         cell->car = NULL;
         free_cell(cell_pool, cell);
         freed++;
       }
     while (--cell >= cell_pool->cells);
   } while (cell_pool = cell_pool->next);
   cell = new_cell(NULL, x_env.x_int);
   set_val(cell, freed);
   return cell;
}