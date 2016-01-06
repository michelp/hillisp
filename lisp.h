#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>

#define X_YOUNG_CELL_POOL_SIZE (1024*64)
#define X_OLD_CELL_POOL_SIZE (1024*512)
#define X_XECTOR_BLOCK_SIZE (1024*1024)

#define X_NUM_FRAMES 128
#define X_HASH_TABLE_SIZE 269
#define X_HASH_MULTIPLIER 131
#define X_MAX_NAME_LEN 128

#define THREADSPERBLOCK 256
#define BLOCKS 256

#define CHECK check_cuda_errors(__FILE__, __LINE__)
#define GDX gridDim.x
#define BDX blockDim.x
#define BIX blockIdx.x
#define TIX threadIdx.x
#define TID ((BDX * BIX) + TIX)
#define STRIDE (BDX * GDX)

typedef struct x_cell x_cell, *x_any;
typedef x_any (*x_fn0)();
typedef x_any (*x_fn1)(x_any);
typedef x_any (*x_fn2)(x_any, x_any);
typedef x_any (*x_fn3)(x_any, x_any, x_any);
typedef x_any (*x_fnv)(x_any);

struct __align__(16) x_cell {
  x_any car;
  x_any cdr;
  x_any type;
  void *value;
};

typedef struct __align__(16) x_cell_pool {
  x_cell cells[X_YOUNG_CELL_POOL_SIZE];
  x_any free;
  struct x_cell_pool *next;
} x_cell_pool;

#define free_cell(h, p) (car(p) = h->free, h->free=(p))

#define current_frame_bucket(h) (x_env.frames[x_env.frame_count][h])
#define get_frame_bucket(i, h) (x_env.frames[i][h])

typedef struct __align__(16) x_environ {
  cudaStream_t stream;
  cudaError_t result;
  int debugLevel;

  x_any symbol;
  x_any binding;
  x_any ns;
  x_any nil;
  x_any true_;
  x_any dot;
  x_any lparen;
  x_any rparen;
  x_any lbrack;
  x_any rbrack;
  x_any eof;
  x_any builtin;
  x_any token;
  x_any user;
  x_any pair;
  x_any int_;
  x_any double_;
  x_any str;
  x_any ixector;
  x_any dxector;
  x_any fn0;
  x_any fn1;
  x_any fn2;
  x_any fn3;
  x_any fnv;
  x_any special;

  x_cell_pool* cell_pools;

  int frame_count;
  int max_frame_count;
  x_any frames[X_NUM_FRAMES][X_HASH_TABLE_SIZE];
} x_environ;

extern __thread x_environ x_env;

#define car(x) (x->car)
#define cdr(x) (x->cdr)
#define cadr(x) (car(cdr(x)))
#define caddr(x) (car(cdr(cdr(x))))
#define cadddr(x) (car(cdr(cdr(cdr(x)))))
#define cddr(x) (cdr(cdr(x)))
#define cdddr(x) (cdr(cdr(cdr(x))))

#define type(x) ((x)->type)
#define val(x) ((x)->value)
#define ival(x) ((int64_t)val(x))
#define fval(x) (*((double*)val(x)))
#define sval(x) ((char*)val(x))
#define xval(x) ((x_any)val(x))

#define set_car(x, y) (car(x) = (y))
#define set_cdr(x, y) (cdr(x) = (y))
#define set_type(x, y) (type(x) = (y))
//#define set_val(x, y) val(x) = ((void*)y)
#define set_val(x, y) ((x->value) = (void*)(y))
#define set_val(x, y) ((x->value) = (void*)(y))

template <typename T> inline T* cars(x_any x) { return (T*)(xval(x)); }

#define xector_size(x) (ival(car(x)))

#define xector_car_ith(x, i) (cars<int64_t>((x))[(i)])

#define xector_car_dth(x, i) (cars<double>((x))[(i)])

#define xector_set_car_ith(x, i, y) (cars<void*>((x))[(i)]) = (void*)(y)

#define is_symbol(x) (type(x) == x_env.symbol)
#define is_token(x) (type(x) == x_env.token)
#define is_user(x) (type(x) == x_env.user)
#define is_pair(x) (type(x) == x_env.pair)
#define is_binding(x) (type(x) == x_env.binding)
#define is_ixector(x) (type(x) == x_env.ixector)
#define is_dxector(x) (type(x) == x_env.dxector)
#define is_xector(x) (is_ixector(x) || is_dxector(x))
#define is_int(x) (type(x) == x_env.int_)
#define is_double(x) (type(x) == x_env.double_)
#define is_str(x) (type(x) == x_env.str)

#define are_symbols(x, y) (is_symbol(x) && is_symbol(y))
#define are_pairs(x, y) (is_pair(x) && is_pair(y))
#define are_xectors(x, y) (is_xector(x) && is_xector(y))
#define are_ixectors(x, y) (is_ixector(x) && is_ixector(y))
#define are_dxectors(x, y) (is_dxector(x) && is_dxector(y))
#define are_ints(x, y) (is_int(x) && is_int(y))
#define are_doubles(x, y) (is_double(x) && is_double(y))
#define are_strs(x, y) (is_str(x) && is_str(y))

#define is_fn0(x) (type(x) == x_env.fn0)
#define is_fn1(x) (type(x) == x_env.fn1)
#define is_fn2(x) (type(x) == x_env.fn2)
#define is_fn3(x) (type(x) == x_env.fn3)
#define is_fnv(x) (type(x) == x_env.fnv)
#define is_special(x) (type(x) == x_env.special)
#define is_builtin(x) (is_fn0(x) || is_fn1(x) || is_fn2(x) || is_fn3(x) || is_fnv(x))
#define is_atom(x) (is_builtin((x)) || is_special(x) || is_user(x) || is_int(x) || is_xector(x))
#define are_atoms(x, y) (is_atom(x) && is_atom(y))
#define is_func(x) (is_builtin((x)) || is_user((x)) || is_special(x))

#define assert_xectors_align(x, y) assert(xector_size(x) == xector_size(y))

// REPL functions

void* x_alloc(size_t);
char* new_name(const char*);
x_any new_cell(const char*, x_any);
x_any new_int(int64_t);
x_any new_double(double);
x_any new_ixector(size_t);
x_any new_dxector(size_t);

x_cell_pool* new_cell_pool(x_cell_pool*);
x_any def_token(const char*);
int hash(const char*);
x_any lookup(const char*, int);
x_any create_symbol(const char*);
char* x_str(x_any);
void inline print_el(FILE*, x_any, int);
void print_list(x_any, FILE*);
void print_cell(x_any, FILE*);
void print_list(x_any, FILE*);
x_any _bind(const char*, x_any, int);
x_any bind(const char*, x_any);
x_any local(const char*, x_any);
x_any intern(const char*);
int64_t length(x_any);
x_any eval_symbol(x_any);
x_any eval_list(x_any);
x_any intern(const char*);
x_any def_builtin(const char*, void*, int);
x_any read_token(FILE*);
x_any read_xector(FILE*);
x_any read_sexpr(FILE*);
x_any read_cdr(FILE*);
x_any read_sexpr_head(FILE*);
x_any read_sexpr_tail(FILE*);
void init(void);
void init_frames();

// core functions

x_any x_car(x_any);
x_any x_cdr(x_any);
x_any x_cons(x_any, x_any);
x_any x_list(x_any);
x_any x_print(x_any);
x_any x_println(x_any);
x_any x_eval(x_any);
x_any x_apply(x_any, x_any);
x_any x_is(x_any, x_any);
x_any x_isinstance(x_any, x_any);
x_any x_assert(x_any);
x_any x_asserteq(x_any, x_any);
x_any x_type(x_any);
x_any x_len(x_any);
x_any x_range(x_any, x_any, x_any);
x_any x_set(x_any);
x_any x_fill(x_any, x_any);
x_any x_dir();

// special

x_any x_quote(x_any);
x_any x_def(x_any);

// flow

x_any x_if(x_any);
x_any x_while(x_any);
x_any x_do(x_any);
x_any x_for(x_any);
x_any x_collect(x_any);

// math

x_any _x_add(x_any, x_any, bool);
x_any x_add(x_any, x_any);
x_any x_addass(x_any, x_any);

x_any _x_sub(x_any, x_any, bool);
x_any x_sub(x_any, x_any);
x_any x_subass(x_any, x_any);

x_any _x_mul(x_any, x_any, bool);
x_any x_mul(x_any, x_any);
x_any x_mulass(x_any, x_any);

x_any _x_div(x_any, x_any, bool);
x_any x_div(x_any, x_any);
x_any x_divass(x_any, x_any);

x_any _x_fma(x_any, x_any, x_any, bool);
x_any x_fma(x_any, x_any, x_any);
x_any x_fmaass(x_any, x_any, x_any);

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

x_any x_gc();
x_any x_time();

template<typename T> __global__ void xd_add(const T* __restrict__, const T* __restrict__, T* __restrict__, const size_t);
template<typename T> __global__ void xd_sub(const T* __restrict__, const T* __restrict__, T* __restrict__, const size_t);
template<typename T> __global__ void xd_mul(const T* __restrict__, const T* __restrict__, T* __restrict__, const size_t);
template<typename T> __global__ void xd_div(const T* __restrict__, const T* __restrict__, T* __restrict__, const size_t);
template<typename T> __global__ void xd_fma(const T* __restrict__, const T* __restrict__, T* __restrict__, const size_t);
template<typename T> __global__ void xd_saxpy(const T* __restrict__, const T* __restrict__, T* __restrict__, const size_t);

template<typename T> __global__ void xd_eq(const T* __restrict__, const T* __restrict__, int64_t* __restrict__, const size_t);
template<typename T> __global__ void xd_all(const T* __restrict__, int* __restrict__, const size_t);
template<typename T> __global__ void xd_any(const T* __restrict__, int* __restrict__, const size_t);
template<typename T> __global__ void xd_fill(T* __restrict__, const T, const size_t);

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

void inline push_frame() {
  x_env.frame_count += 1;
  x_env.max_frame_count += 1;
  for (int i = 0; i < X_HASH_TABLE_SIZE; i++)
    current_frame_bucket(i) = x_env.nil;
}

void inline pop_frame() {
  x_env.frame_count -= 1;
}

