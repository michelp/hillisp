#include <stdint.h>
#include <stdarg.h>

typedef struct x_cell x_cell, *x_any;

typedef x_any (*x_fn0_t)();
typedef x_any (*x_fn1_t)(x_any);
typedef x_any (*x_fn2_t)(x_any, x_any);
typedef x_any (*x_fn3_t)(x_any, x_any, x_any);


#define X_HEAP_BLOCK_SIZE (1024*1024/sizeof(x_cell))
#define X_XECTOR_BLOCK_SIZE (256*1024/sizeof(void*))

struct __align__(16) x_cell {
  void *car;
  void *cdr;
  char *name;
  x_any type;
};

struct __align__(16) x_xector_t {
  void *cars[X_XECTOR_BLOCK_SIZE];
  void *cdrs[X_XECTOR_BLOCK_SIZE];
  char *names[X_XECTOR_BLOCK_SIZE];
  uint64_t types[X_XECTOR_BLOCK_SIZE];
  size_t size;
  struct x_xector_t *next;
} x_xector_t;

typedef struct __align__(16) x_heap {
  x_cell cells[X_HEAP_BLOCK_SIZE];
  size_t used;
  struct x_heap *next;
} x_heap;

#define car(x) ((x_any)(x)->car)
#define cdr(x) ((x_any)(x)->cdr)
#define int_car(x) (int64_t)((x_any)(x)->car)
#define int_cdr(x) (int64_t)((x_any)(x)->cdr)

#define set_car(x, y) ((x)->car) = (void*)(y)
#define set_cdr(x, y) ((x)->cdr) = (void*)(y)

#define type(x) ((x)->type)
#define name(x) ((x)->name)
#define size(x) ((x)->size)


#define is_symbol(x) ((type(x) == x_symbol) || is_int(x))
#define is_token(x) (type(x) == x_builtin)
#define is_user(x) (type(x) == x_user)
#define is_pair(x) (type(x) == x_pair)
#define is_xector(x) (type(x) == x_xector)

#define is_int(x) (type(x) == x_int)

#define is_fn0(x) (type(x) == x_fn0)
#define is_fn1(x) (type(x) == x_fn1)
#define is_fn2(x) (type(x) == x_fn2)
#define is_fn3(x) (type(x) == x_fn3)

#define is_builtin(x) (is_fn0(x) || is_fn1(x) || is_fn2(x) || is_fn3(x))

#define is_atom(x) (is_symbol((x)) || is_builtin((x)) || is_token((x)))
#define is_func(x) (is_builtin((x)) || is_user((x)))

#define X_HASH_TABLE_SIZE 269
#define X_HASH_MULTIPLIER 131
#define X_MAX_NAME_LEN 128
typedef x_any hash_table_type[X_HASH_TABLE_SIZE];


// REPL functions

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
void init(void);

// core functions

x_any x_car(x_any);
x_any x_cdr(x_any);
x_any x_cons(x_any, x_any);
x_any x_print(x_any);
x_any x_eval(x_any);
x_any x_apply(x_any, x_any);
x_any x_quote(x_any);
x_any x_cond(x_any);
x_any x_is(x_any, x_any);
x_any x_isinstance(x_any, x_any);

// math

x_any x_add(x_any, x_any);
x_any x_sub(x_any, x_any);
x_any x_mul(x_any, x_any);
x_any x_div(x_any, x_any);

// cmp

x_any x_eq(x_any, x_any);
x_any x_neq(x_any, x_any);
x_any x_lt(x_any, x_any);
x_any x_gt(x_any, x_any);

// bool

x_any x_not(x_any);
x_any x_and(x_any, x_any);
x_any x_or(x_any, x_any);


