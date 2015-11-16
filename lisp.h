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
  void *device;
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
#define device(x) ((x)->device)
#define size(x) ((x)->size)

#define is_symbol(x) (flags(x) & SYMBOL)
#define is_builtin(x) (flags(x) & BUILTIN)
#define is_token(x) (flags(x) & BUILTIN)
#define is_atom(x) (is_symbol((x)) || is_builtin((x)) || is_token((x)))

#define is_user(x) (flags(x) & USER)
#define is_pair(x) (flags(x) & PAIR)
#define is_xector(x) (flags(x) & XECTOR)

#define HASH_TABLE_SIZE	269
#define HASH_MULTIPLIER	131
#define MAX_NAME_LEN	128
typedef x_any hash_table_type[HASH_TABLE_SIZE];

char* new_name(const char*);
x_any new_cell(const char*);
x_any def_token(const char*);
int hash(const char*);
x_any lookup(const char*, x_any);
x_any create_symbol(const char*);
void print_list(x_any, FILE*);
void print_cell(x_any, FILE*);
void print_list(x_any, FILE*);
void enter(x_any);
x_any intern(const char*);
int length(x_any);
x_any list_eval(x_any);
x_any intern(const char*);
x_any def_builtin(const char*, void*, size_t);
x_any read_token(FILE*);
x_any read_sexpr(FILE*);
x_any read_cdr(FILE*);
x_any read_head(FILE*);
x_any read_tail(FILE*);

x_any x_car(x_any);
x_any x_cdr(x_any);
x_any x_cons(x_any, x_any);
x_any x_print(x_any);
x_any x_eval(x_any);
x_any x_apply(x_any, x_any);
x_any x_quote(x_any);
x_any x_cond(x_any);
x_any x_is(x_any, x_any);

void init(void);
