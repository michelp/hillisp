#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <inttypes.h>
#include "lisp.h"

cudaStream_t stream;
cudaError_t result;

//#define DEBUG 1

#ifdef DEBUG
int debugLevel = 0;
#endif

x_any x_symbol;
x_any x_garbage;
x_any x_nil;
x_any x_true;
x_any x_dot;
x_any x_lparen;
x_any x_rparen;
x_any x_lbrack;
x_any x_rbrack;
x_any x_eof;
x_any x_builtin;
x_any x_token;
x_any x_user;
x_any x_pair;
x_any x_xector;
x_any x_int;
x_any x_fn0;
x_any x_fn1;
x_any x_fn2;
x_any x_fn3;
hash_table_type hash_table;

__device__ __host__ void* bi_malloc(size_t size) {
void* result;
#ifdef __CUDA_ARCH__
  result = malloc(size);
#else
  cudaMallocManaged(&result, size);
  cudaStreamAttachMemAsync(stream, result);
  SYNCS(stream);
  CHECK;
#endif
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
  cell = (x_any)malloc(sizeof(x_cell));
  set_cdr(cell, NULL);
  set_car(cell, NULL);
  type(cell) = type;
  if (name == NULL)
    name(cell) = NULL;
  else
    name(cell) = new_name(name);
  return cell;
}

x_any new_xector(const char* name, size_t size) {
  x_any cell;
  x_any_x xector;
  cell = new_cell(name, x_xector);
  xector = (x_any_x)malloc(sizeof(x_xector_t));
  xector->cars = (void**)bi_malloc(size * sizeof(void*));
  xector->size = size;
  set_cdr(cell, xector);
  return cell;
}

int hash(const char *name) {
  int value = 0;
  while (*name != '\0')
    value = (value * X_HASH_MULTIPLIER + *name++) % X_HASH_TABLE_SIZE;
  return value;
}

x_any lookup(const char *name, x_any cell) {
  if (cell == x_nil)
    return NULL;
  else if (strcmp(name(car<x_any>(cell)), name) == 0)
    return car<x_any>(cell);
  else
    return lookup(name, cdr<x_any>(cell));
}

x_any create_symbol(const char *new_name) {
  x_any cell;
  cell = new_cell(new_name, x_symbol);
  if (isdigit(new_name[0]) || (new_name[0] == '-' && isdigit(new_name[1]))) {
    set_car(cell, atol(new_name));
    type(cell) = x_int;
  }
  return cell;
}

void enter(x_any cell) {
  int hash_val;
  hash_val = hash(name(cell));
  hash_table[hash_val] = x_cons(cell, hash_table[hash_val]);
}

x_any intern(const char *name) {
  x_any cell;
  cell = lookup(name, hash_table[hash(name)]);
  if (cell != NULL)
    return cell;
  else {
    cell = create_symbol(name);
    enter(cell);
    return cell;
  }
}

int length(x_any cell) {
  if (cell == x_nil)
    return 0;
  else
    return 1 + length(cdr<x_any>(cell));
}

x_any list_eval(x_any cell) {
  if (cell == x_nil)
    return x_nil;
  if (is_atom(cell))
    return cell;
  else
    return x_cons(x_eval(car<x_any>(cell)), list_eval(cdr<x_any>(cell)));
}

x_any read_token(FILE *infile) {
  int c;
  static char buf[X_MAX_NAME_LEN];
  char *ptr = buf;

  do {
    c = getc(infile);
    if (c == ';')
      do c = getc(infile); while (c != '\n' && c != EOF);
  } while (isspace(c));
  switch (c) {
  case EOF:
    return x_eof;
  case '(':
    return x_lparen;
  case ')':
    return x_rparen;
  case '[':
    return x_lbrack;
  case ']':
    return x_rbrack;
  case '.':
    return x_dot;
  default:
    *ptr++ = c;
    while ((c = getc(infile)) != EOF &&
           !isspace(c) &&
           c != '(' && c != ')' &&
           c != '[' && c!= ']')
      *ptr++ = c;
    if (c != EOF)
      ungetc(c, infile);
    *ptr = '\0';
    if (strcmp(buf, "symbol") == 0)
      return x_symbol;
    return intern(buf);
  }
}

x_any read_cdr(FILE *infile) {
  x_any cdr;
  x_any token;
  cdr = read_sexpr(infile);
  token = read_token(infile);
  if (token == x_rparen)
    return cdr;
  else
    assert(0);
  return x_nil;
}

x_any read_sexpr_tail(FILE *infile) {
  x_any token;
  x_any temp;
  token = read_token(infile);
  if (is_symbol(token) || is_builtin(token))
    return x_cons(token, read_sexpr_tail(infile));
  if (token == x_lparen) {
    temp = read_sexpr_head(infile);
    return x_cons(temp, read_sexpr_tail(infile));
  }
  if (token == x_lbrack) {
    temp = read_xector(infile);
    return x_cons(temp, read_sexpr_tail(infile));
  }
  if (token == x_dot)
    return read_cdr(infile);
  if (token == x_rparen)
    return x_nil;
  if (token == x_eof)
    assert(0);
  return x_nil;
}

x_any read_sexpr_head(FILE *infile) {
  x_any token;
  x_any temp;
  token = read_token(infile);
  if (is_symbol(token) || is_builtin(token))
    return x_cons(token, read_sexpr_tail(infile));
  else if (token == x_lparen) {
    temp = read_sexpr_head(infile);
    return x_cons(temp, read_sexpr_tail(infile));
  }
  else if (token == x_lbrack) {
    temp = read_xector(infile);
    return x_cons(temp, read_sexpr_tail(infile));
  }
  else if (token == x_rparen)
    return x_nil;
  else if (token == x_dot)
    assert(0);
  else if (token == x_eof)
    assert(0);
  return x_nil;
}

x_any read_xector(FILE *infile) {
  x_any val;
  x_any cell;
  x_any typ = NULL;
  size_t size = 0;
  cell = new_xector("xector", X_XECTOR_BLOCK_SIZE);
  do {
    val = x_eval(read_sexpr(infile));
    if (val == x_nil)
      break;
    if (typ == NULL)
      typ = type(val);
    else if (type(val) != typ)
      assert(0); // must all be same type

    if (typ == x_int)
      xector_set_car_ith(cell, size, car<int64_t>(val));
    else if (typ == x_xector)
      xector_set_car_ith(cell, size, car<x_any>(val));
    else
      assert(0);
    size++;
  } while (1);
  xector_size(cell) = size;
  return cell;
}

x_any read_sexpr(FILE *infile) {
  x_any token;
  token = read_token(infile);
  if (is_symbol(token) || is_builtin(token))
    return token;
  if (token == x_lbrack)
    return read_xector(infile);
  if (token == x_lparen)
    return read_sexpr_head(infile);
  if (token == x_rparen)
    assert(0);
  if (token == x_dot)
    assert(0);
  if (token == x_eof)
    return token;
  return x_nil;
}

x_any def_token(const char* new_name) {
  return new_cell(new_name, x_token);
}

x_any def_builtin(char const *name, void *fn, size_t num_args, void *dfn) {
  x_any cell;
  cell = intern(name);
  type(cell) = x_builtin;
  set_cdr(cell, fn);
  set_car(cell, dfn);
  switch(num_args) {
  case 0:
    type(cell) = x_fn0;
    break;
  case 1:
    type(cell) = x_fn1;
    break;
  case 2:
    type(cell) = x_fn2;
    break;
  case 3:
    type(cell) = x_fn3;
    break;
  }
  return cell;
}

void init(void) {
  x_symbol = new_cell("symbol", NULL);
  type(x_symbol) = x_symbol;
  x_pair = new_cell("pair", NULL);
  enter(x_symbol);
  enter(x_pair);

  x_nil = create_symbol("nil");
  for (int i = 0; i < X_HASH_TABLE_SIZE; i++)
    hash_table[i] = x_nil;
  enter(x_nil);

  x_garbage = intern("garbage");
  x_token = intern("token");
  x_builtin = intern("builtin");
  x_user = intern("user");
  x_true = intern("true");
  x_xector = intern("xector");
  x_int = intern("int");
  x_fn0 = intern("fn0");
  x_fn1 = intern("fn1");
  x_fn2 = intern("fn2");
  x_fn3 = intern("fn3");

  x_dot = def_token(".");
  x_lparen = def_token("(");
  x_rparen = def_token(")");
  x_lbrack = def_token("[");
  x_rbrack = def_token("]");
  x_eof = def_token("EOF");

  def_builtin("is", (void*)x_is, 2, NULL);
  def_builtin("isinstance", (void*)x_isinstance, 2, NULL);
  def_builtin("type", (void*)x_type, 1, NULL);
  def_builtin("car", (void*)x_car, 1, NULL);
  def_builtin("cdr", (void*)x_cdr, 1, NULL);
  def_builtin("cons", (void*)x_cons, 2, NULL);
  def_builtin("quote", (void*)x_quote, 1, NULL);
  def_builtin("if", (void*)x_if, 1, NULL);
  def_builtin("while", (void*)x_while, 1, NULL);
  def_builtin("eval", (void*)x_eval, 1, NULL);
  def_builtin("apply", (void*)x_apply, 2, NULL);
  def_builtin("assert", (void*)x_assert, 1, NULL);
  def_builtin("print", (void*)x_print, 1, NULL);
  def_builtin("println", (void*)x_println, 1, NULL);
  def_builtin("+", (void*)x_add, 2, NULL);
  def_builtin("-", (void*)x_sub, 2, NULL);
  def_builtin("*", (void*)x_mul, 2, NULL);
  def_builtin("/", (void*)x_div, 2, NULL);
  def_builtin("fma", (void*)x_fma, 3, NULL);
  def_builtin("==", (void*)x_eq, 2, NULL);
  def_builtin("!=", (void*)x_neq, 2, NULL);
  def_builtin(">", (void*)x_gt, 2, NULL);
  def_builtin("<", (void*)x_lt, 2, NULL);
  def_builtin("not", (void*)x_not, 1, NULL);
  def_builtin("and", (void*)x_and, 2, NULL);
  def_builtin("all", (void*)x_all, 1, NULL);
  def_builtin("any", (void*)x_any_, 1, NULL);
  def_builtin("or", (void*)x_or, 2, NULL);
  def_builtin("fill", (void*)x_fill, 2, NULL);
  def_builtin("time", (void*)x_time, 0, NULL);
}

int main(int argc, const char* argv[]) {
  x_any expr;
  x_any value;
  FILE *fp;
  result = cudaStreamCreate(&stream);

  init();
  if (argc > 1) {
    for (int i = 1; i < argc; i++) {
      if (argv[i][0] == '-') {
        continue;
      }
      else {
        fp = fopen(argv[i], "r");
        if (fp == NULL)
          assert(0);
        for (;;) {
          expr = read_sexpr(fp);
          if (expr == x_eof)
            break;
          value = x_eval(expr);
        }
      }
    }
  }
  else {
    for (;;) {
      printf("? ");
      expr = read_sexpr(stdin);
      if (expr == x_eof)
        break;
      value = x_eval(expr);
      printf(": ");
      print_cell(value, stdout);
      putchar('\n');
    }
  }
  result = cudaStreamDestroy(stream);
  cudaDeviceReset();
}
