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

  for (int i = x_env.max_frame_count; i >= 0; i--)
    for (int j = 0; j < X_HASH_TABLE_SIZE; j++)
      mark(x_env.frames[i][j]);

  x_env.max_frame_count = x_env.frame_count;

   cell_pool = x_env.cell_pools;
   SYNCS(x_env.stream);
   do {
     cell = cell_pool->cells + X_YOUNG_CELL_POOL_SIZE-1;
     do
       if ((int64_t)(cell->cdr) & 1) {
         if (is_xector(cell)) {
           if (xval(cell) != NULL) {
             cudaFree(xval(cell));
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
   return new_int(freed);
}

x_any c_alloc(x_any type) {
  x_any cell;
  if (!(cell = x_env.cell_pools->free))
    assert(0);
  x_env.cell_pools->free = car(cell);
  set_cdr(cell, NULL);
  set_car(cell, NULL);
  set_type(cell, type);
  return cell;
}

void* x_alloc(size_t size) {
  void* result;
  cudaMallocManaged(&result, size);
  cudaStreamAttachMemAsync(x_env.stream, result);
  CHECK;
  assert(result != NULL);
  return result;
}

char* new_name(const char* name) {
  char *n;
  n = (char*)malloc(strlen(name) + 1);
  strcpy(n, name);
  return n;
}

x_any new_cell(const char* name, x_any type) {
  x_any cell;
  cell = c_alloc(type);
  if (name == NULL)
    set_val(cell, NULL);
  else
    set_val(cell, new_name(name));
  return cell;
}

x_any new_int(int64_t value) {
  x_any cell;
  cell = new_cell(NULL, x_env.int_);
  set_val(cell, value);
  return cell;
}

x_any new_float(double value) {
  x_any cell;
  float *v;
  cell = new_cell(NULL, x_env.float_);
  v = (float*)malloc(sizeof(float));
  assert(v != NULL);
  *v = value;
  set_val(cell, v);
  return cell;
}

x_cell_pool* new_cell_pool(x_cell_pool* old) {
  x_cell_pool* h;
  x_any cell;
  h = (x_cell_pool*)malloc(sizeof(x_cell_pool));
  h->next = old;
  cell = h->cells + X_YOUNG_CELL_POOL_SIZE - 1;
  do
    free_cell(h, cell);
  while (--cell >= h->cells);
  return h;
}

void init_frames() {
  for (int i = 0; i < X_NUM_FRAMES; i++)
    for (int j = 0; j < X_HASH_TABLE_SIZE; j++)
      x_env.frames[i][j] = x_env.nil;

  x_env.frame_count = 0;
  x_env.max_frame_count = 0;
}