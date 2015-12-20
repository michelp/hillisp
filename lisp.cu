#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <inttypes.h>
#include "lisp.h"

__thread x_environ x_env;

x_any c_alloc(x_any type) {
  x_any cell;
  if (!(cell = x_env.x_heaps->free))
    assert(0);
  x_env.x_heaps->free = car(cell);
  set_cdr(cell, NULL);
  set_car(cell, NULL);
  type(cell) = type;
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

x_heap* new_heap(x_heap* old) {
  x_heap* h;
  x_any cell;
  h = (x_heap*)malloc(sizeof(x_heap));
  h->next = old;
  cell = h->cells + X_YOUNG_HEAP_SIZE - 1;
  do
    free_cell(h, cell);
  while (--cell >= h->cells);
  return h;
}

x_frame* new_frame() {
  x_frame * f;
  f = (x_frame*)malloc(sizeof(x_frame));
  f->next = NULL;
  f->prev = NULL;
  for (int i = 0; i < X_HASH_TABLE_SIZE; i++)
    f->names[i] = x_env.x_nil;
  return f;
}

template<typename T>
x_any new_xector(const char* name, size_t size) {
  x_any cell;
  x_any_x xector;
  cell = new_cell(name, x_env.x_xector);
  xector = (x_any_x)malloc(sizeof(x_xector_t));
  xector->cars = (void**)x_alloc(size * sizeof(T));
  xector->size = size;
  set_val(cell, xector);
  return cell;
}

int length(x_any cell) {
  if (cell == x_env.x_nil)
    return 0;
  else
    return 1 + length(cdr(cell));
}

x_any list_eval(x_any cell) {
  if (cell == x_env.x_nil)
    return x_env.x_nil;
  if (is_atom(cell))
    return cell;
  else
    return x_cons(x_eval(car(cell)), list_eval(cdr(cell)));
}

x_any read_token(FILE *infile) {
  int c;
  static char buf[X_MAX_NAME_LEN];
  char *ptr = buf;
  x_any cell;

  do {
    c = getc(infile);
    if (c == ';')
      do c = getc(infile); while (c != '\n' && c != EOF);
  } while (isspace(c));
  switch (c) {
  case EOF:
    return x_env.x_eof;
  case '(':
    return x_env.x_lparen;
  case ')':
    return x_env.x_rparen;
  case '[':
    return x_env.x_lbrack;
  case ']':
    return x_env.x_rbrack;
  case '.':
    return x_env.x_dot;
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
      return x_env.x_symbol;

    if (isdigit(buf[0]) || (buf[0] == '-' && isdigit(buf[1]))) {
      cell = c_alloc(x_env.x_int);
      set_val(cell, atoll(buf));
      return cell;
    }
    return intern(buf);
  }
}

x_any read_cdr(FILE *infile) {
  x_any cdr;
  x_any token;
  cdr = read_sexpr(infile);
  token = read_token(infile);
  if (token == x_env.x_rparen)
    return cdr;
  else
    assert(0);
  return x_env.x_nil;
}

x_any read_sexpr_tail(FILE *infile) {
  x_any token;
  x_any temp;
  token = read_token(infile);
  if (is_atom(token))
    return x_cons(token, read_sexpr_tail(infile));
  if (token == x_env.x_lparen) {
    temp = read_sexpr_head(infile);
    return x_cons(temp, read_sexpr_tail(infile));
  }
  if (token == x_env.x_lbrack) {
    temp = read_xector(infile);
    return x_cons(temp, read_sexpr_tail(infile));
  }
  if (token == x_env.x_dot)
    return read_cdr(infile);
  if (token == x_env.x_rparen)
    return x_env.x_nil;
  if (token == x_env.x_eof)
    assert(0);
  return x_env.x_nil;
}

x_any read_sexpr_head(FILE *infile) {
  x_any token;
  x_any temp;
  token = read_token(infile);
  if (is_atom(token))
    return x_cons(token, read_sexpr_tail(infile));
  else if (token == x_env.x_lparen) {
    temp = read_sexpr_head(infile);
    return x_cons(temp, read_sexpr_tail(infile));
  }
  else if (token == x_env.x_lbrack) {
    temp = read_xector(infile);
    return x_cons(temp, read_sexpr_tail(infile));
  }
  else if (token == x_env.x_rparen)
    return x_env.x_nil;
  else if (token == x_env.x_dot)
    assert(0);
  else if (token == x_env.x_eof)
    assert(0);
  return x_env.x_nil;
}

x_any read_xector(FILE *infile) {
  x_any val;
  x_any cell;
  x_any typ = NULL;
  size_t size = 0;
  cell = new_xector<int64_t>("xector", X_XECTOR_BLOCK_SIZE);
  do {
    val = x_eval(read_sexpr(infile));
    if (val == x_env.x_nil)
      break;
    if (typ == NULL)
      typ = type(val);
    else if (type(val) != typ)
      assert(0); // must all be same type

    if (typ == x_env.x_int)
      xector_set_car_ith(cell, size, ival(val));
    else if (typ == x_env.x_xector)
      xector_set_car_ith(cell, size, car(val));
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
  if (is_atom(token))
    return token;
  if (token == x_env.x_lbrack)
    return read_xector(infile);
  if (token == x_env.x_lparen)
    return read_sexpr_head(infile);
  if (token == x_env.x_rparen)
    assert(0);
  if (token == x_env.x_dot)
    assert(0);
  if (token == x_env.x_eof)
    return token;
  return x_env.x_nil;
}

x_any def_token(const char* new_name) {
  return new_cell(new_name, x_env.x_token);
}

x_any def_builtin(char const *name, void *fn, size_t num_args, void *dfn) {
  x_any cell;
  cell = intern(name);
  type(cell) = x_env.x_builtin;
  set_car(cell, (x_any)fn);
  switch(num_args) {
  case 0:
    type(cell) = x_env.x_fn0;
    break;
  case 1:
    type(cell) = x_env.x_fn1;
    break;
  case 2:
    type(cell) = x_env.x_fn2;
    break;
  case 3:
    type(cell) = x_env.x_fn3;
    break;
  }
  return cell;
}

void init(void) {
  x_env.x_heaps = new_heap(NULL);

  x_env.x_symbol = new_cell("symbol", NULL);
  type(x_env.x_symbol) = x_env.x_symbol;
  x_env.x_pair = new_cell("pair", NULL);
  x_env.x_nil = new_cell("nil", x_env.x_symbol);

  x_env.x_frames = new_frame();

  bind("nil", x_env.x_nil, x_env.x_frames);
  bind("symbol", x_env.x_symbol, x_env.x_frames);
  bind("pair", x_env.x_pair, x_env.x_frames);

  x_env.x_binding = intern("binding");
  x_env.x_token = intern("token");
  x_env.x_builtin = intern("builtin");
  x_env.x_user = intern("user");
  x_env.x_true = intern("true");
  x_env.x_xector = intern("xector");
  x_env.x_int = intern("int");
  x_env.x_str = intern("str");

  x_env.x_fn0 = intern("fn0");
  x_env.x_fn1 = intern("fn1");
  x_env.x_fn2 = intern("fn2");
  x_env.x_fn3 = intern("fn3");

  x_env.x_dot = def_token(".");
  x_env.x_lparen = def_token("(");
  x_env.x_rparen = def_token(")");
  x_env.x_lbrack = def_token("[");
  x_env.x_rbrack = def_token("]");
  x_env.x_eof = def_token("EOF");

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
  def_builtin("gc", (void*)x_gc, 0, NULL);
  def_builtin("set", (void*)x_set, 2, NULL);
}

int main(int argc, const char* argv[]) {
  x_any expr;
  x_any value;
  FILE *fp;
  x_env.result = cudaStreamCreate(&x_env.stream);

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
          if (expr == x_env.x_eof)
            break;
          value = x_eval(expr);
          x_gc();
        }
      }
    }
  }
  else {
    for (;;) {
      printf("? ");
      expr = read_sexpr(stdin);
      if (expr == x_env.x_eof)
        break;
      value = x_eval(expr);
      printf(": ");
      print_cell(value, stdout);
      putchar('\n');
      x_gc();
    }
  }
  x_env.result = cudaStreamDestroy(x_env.stream);
  cudaDeviceReset();
}
