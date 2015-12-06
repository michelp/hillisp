#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdarg.h>

#define X_HEAP_BLOCK_SIZE (1024*1024/sizeof(x_cell))
#define X_XECTOR_BLOCK_SIZE (256*1024/sizeof(void*))

typedef struct x_cell x_cell, *x_any;
typedef x_any (*x_fn0_t)();
typedef x_any (*x_fn1_t)(x_any);
typedef x_any (*x_fn2_t)(x_any, x_any);
typedef x_any (*x_fn3_t)(x_any, x_any, x_any);

struct __align__(16) x_cell {
  void *car;
  void *cdr;
  char *name;
  x_any type;
};

typedef struct __align__(16) x_xector_t {
  void **cars;
  void **cdrs;
  char **names;
  uint64_t **types;
  size_t size;
  struct x_xector_t *next;
} x_xector_t, *x_any_x;

typedef struct __align__(16) x_heap {
  x_cell cells[X_HEAP_BLOCK_SIZE];
  size_t used;
  struct x_heap *next;
} x_heap;

template <typename T> T __inline__ carr(x_any x) { return (T)(x->car); }
template <typename T> T __inline__ cdrr(x_any x) { return (T)(x->cdr); }
template <typename T> T __inline__ cadr(x_any x) { return carr<T>(cdrr<T>(x)); }
template <typename T> T __inline__ caddr(x_any x) { return carr<T>(cdrr<T>(cdrr<T>(x))); }
template <typename T> T __inline__ cddr(x_any x) { return cdrr<T>(cdrr<T>(x)); }

#define set_car(x, y) ((x)->car) = (void*)(y)
#define set_cdr(x, y) ((x)->cdr) = (void*)(y)

#define copy_cell(x, y) do {set_car(y, car(x)); set_cdr(y, cdr(x));} while(0)

#define type(x) ((x)->type)
#define name(x) ((x)->name)
#define size(x) ((x)->size)

#define xector_size(x) (((x_any_x)cdrr<x_any>(x))->size)

template <typename T> T* carrs(x_any x) { return (T*)(((x_any_x)cdrr<x_any>(x))->cars); }

#define int64_cdrs(x) ((int64_t*)(((x_any_x)cdr((x_any)(x)))->cdrs))

#define xector_car_ith(x, i) (carrs<int64_t>((x))[(i)])
#define xector_cdr_ith(x, i) ((int64_t)(cdrs((x))[(i)]))

#define xector_set_car_ith(x, i, y) (carrs<void*>((x))[(i)]) = (void*)(y)
#define xector_set_cdr_ith(x, i, y) (cdrs((x))[(i)]) = (void*)(y)

#define is_symbol(x) ((type(x) == x_symbol) || is_int(x))
#define is_token(x) (type(x) == x_builtin)
#define is_user(x) (type(x) == x_user)
#define is_pair(x) (type(x) == x_pair)
#define are_pairs(x, y) (is_pair(x) && is_pair(y))
#define is_xector(x) (type(x) == x_xector)
#define are_xectors(x, y) (is_xector(x) && is_xector(y))
#define xectors_align(x, y) assert(xector_size(x) == xector_size(y))

#define is_int(x) (type(x) == x_int)
#define are_ints(x, y) (is_int(x) && is_int(y))

#define is_fn0(x) (type(x) == x_fn0)
#define is_fn1(x) (type(x) == x_fn1)
#define is_fn2(x) (type(x) == x_fn2)
#define is_fn3(x) (type(x) == x_fn3)

#define is_builtin(x) (is_fn0(x) || is_fn1(x) || is_fn2(x) || is_fn3(x))

#define is_atom(x) (is_symbol((x)) || is_builtin((x)) || is_token((x)) || is_xector(x))
#define are_atoms(x, y) (is_atom(x) && is_atom(y))
#define is_func(x) (is_builtin((x)) || is_user((x)))

#define X_HASH_TABLE_SIZE 269
#define X_HASH_MULTIPLIER 131
#define X_MAX_NAME_LEN 128
typedef x_any hash_table_type[X_HASH_TABLE_SIZE];

#define CHECK check_cuda_errors(__FILE__, __LINE__)
#define GDX gridDim.x
#define BDX blockDim.x
#define BIX blockIdx.x
#define TIX threadIdx.x

#define TID ((BDX * BIX) + TIX)
#define STRIDE (BDX * GDX)
#define TIDE (TID * STRIDE)

#define THREADSPERBLOCK 256
#define BLOCKS 256

#define GRIDBLOCKS(size) ((size) + THREADSPERBLOCK - 1 / THREADSPERBLOCK)

// REPL functions

__device__ __host__ void* bi_malloc(size_t);
char* new_name(const char*);
x_any new_cell(const char*, x_any);
x_any new_xector(const char*, size_t size);
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
x_any read_xector(FILE*);
x_any read_sexpr(FILE*);
x_any read_cdr(FILE*);
x_any read_sexpr_head(FILE*);
x_any read_sexpr_tail(FILE*);
void init(void);

// core functions

x_any x_car(x_any);
x_any x_cdr(x_any);
x_any x_cons(x_any, x_any);
x_any x_print(x_any);
x_any x_println(x_any);
x_any x_eval(x_any);
x_any x_apply(x_any, x_any);
x_any x_quote(x_any);
x_any x_is(x_any, x_any);
x_any x_isinstance(x_any, x_any);
x_any x_assert(x_any);
x_any x_type(x_any);
x_any x_fill(x_any, x_any);

// flow

x_any x_if(x_any);
x_any x_while(x_any, x_any);

// math

x_any x_add(x_any, x_any);
x_any x_sub(x_any, x_any);
x_any x_mul(x_any, x_any);
x_any x_div(x_any, x_any);

x_any x_fma(x_any, x_any, x_any);

// cmp

x_any x_eq(x_any, x_any);
x_any x_neq(x_any, x_any);
x_any x_lt(x_any, x_any);
x_any x_gt(x_any, x_any);

// bool

x_any x_not(x_any);
x_any x_and(x_any, x_any);
x_any x_or(x_any, x_any);
x_any x_all(x_any);
x_any x_any_(x_any);

// xector

x_any x_zeros(x_any);
x_any x_ones(x_any);

// sys

x_any x_time();

extern x_any x_symbol;
extern x_any x_garbage;
extern x_any x_nil;
extern x_any x_true;
extern x_any x_dot;
extern x_any x_lparen;
extern x_any x_rparen;
extern x_any x_lbrack;
extern x_any x_rbrack;
extern x_any x_eof;
extern x_any x_builtin;
extern x_any x_token;
extern x_any x_user;
extern x_any x_pair;
extern x_any x_xector;
extern x_any x_int;
extern x_any x_fn0;
extern x_any x_fn1;
extern x_any x_fn2;
extern x_any x_fn3;
extern hash_table_type hash_table;

template<typename T> __global__ void xd_add(T*, T*, T*, size_t);
template<typename T> __global__ void xd_sub(T*, T*, T*, size_t);
template<typename T> __global__ void xd_mul(T*, T*, T*, size_t);
template<typename T> __global__ void xd_div(T*, T*, T*, size_t);
template<typename T> __global__ void xd_fma(T*, T*, T*, size_t);
template<typename T> __global__ void xd_eq(T*, T*, T*, size_t);
template<typename T> __global__ void xd_all(T*, int*, size_t);
template<typename T> __global__ void xd_any(T*, int*, size_t);
template<typename T> __global__ void xd_fill(T*, T val, size_t);

extern cudaStream_t stream;
extern cudaError_t result;

#define SYNC cudaThreadSynchronize()
#define SYNCS(s) cudaStreamSynchronize(s)

inline void check_cuda_errors(const char *filename, const int line_number)
{
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
    exit(-1);
  }
}
