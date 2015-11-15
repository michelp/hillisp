#include <stdint.h>
#include <stdarg.h>

typedef enum {
  MARK =  1 << 0,

  SYMBOL =  1 << 1,
  BUILTIN = 1 << 2,
  TOKEN = 1 << 3,
  USER = 1 << 4,
  PAIR = 1 << 5,   
  XECTOR = 1 << 6,

  INT = 1 << 7,
  FLOAT = 1 << 8,
  DOUBLE = 1 << 9,

  X_INT = 1 << 10,
  X_FLOAT = 1 << 11,
  X_DOUBLE = 1 << 12

  // syntax
} x_flags;

typedef struct x_xector x_xector;
typedef struct x_cell x_cell, *x_any;
typedef x_any (*x_fn0)();
typedef x_any (*x_fn1)(x_any);
typedef x_any (*x_fn2)(x_any, x_any);
typedef x_any (*x_fn3)(x_any, x_any, x_any);


#define HEAPSIZE (1*1024*1024/sizeof(x_cell)) // Heap allocation unit 1MB

struct __align__(16) x_cell {
  x_any car;
  x_any cdr;
  void *data;
  char *name;
  size_t size;
  x_flags flags;
};

typedef struct __align__(16) x_heap {
  x_cell cells[HEAPSIZE];
  size_t used;
  struct x_heap *next;
} x_heap;

#define car(x) ((x)->car)
#define cdr(x) ((x)->cdr)
#define flags(x) ((x)->flags)
#define name(x) ((x)->name)
#define data(x) ((x)->data)
#define size(x) ((x)->size)

#define is_symbol(x) ((x)->flags & SYMBOL)
#define is_builtin(x) ((x)->flags & BUILTIN)
#define is_token(x) ((x)->flags & BUILTIN)
#define is_atom(x) (is_symbol((x)) || is_builtin((x)) || is_token((x)))

#define is_user(x) ((x)->flags & USER)
#define is_pair(x) ((x)->flags & PAIR)
#define is_xector(x) ((x)->flags & XECTOR)

// never use in host functions
#define x_int ((uint64_t*)(x)->xector.data)
#define x_float ((float*)(x)->xector.data)
#define x_double ((double*)(x)->xector.data)

#define HASH_TABLE_SIZE	269
#define HASH_MULTIPLIER	131
#define MAX_NAME_LEN	128
typedef x_any hash_table_type[HASH_TABLE_SIZE];
