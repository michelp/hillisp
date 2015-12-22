#include "lisp.h"

__thread x_environ x_env;

x_any def_token(const char* new_name) {
  return new_cell(new_name, x_env.token);
}

x_any def_builtin(char const *name, void *fn, size_t num_args, void *dfn) {
  x_any cell;
  cell = intern(name);
  type(cell) = x_env.builtin;
  set_val(cell, fn);
  switch(num_args) {
  case 0:
    type(cell) = x_env.fn0;
    break;
  case 1:
    type(cell) = x_env.fn1;
    break;
  case 2:
    type(cell) = x_env.fn2;
    break;
  case 3:
    type(cell) = x_env.fn3;
    break;
  }
  return cell;
}

void init(void) {
  x_env.cell_pools = new_cell_pool(NULL);

  x_env.symbol = new_cell("symbol", NULL);
  type(x_env.symbol) = x_env.symbol;
  x_env.pair = new_cell("pair", NULL);
  x_env.nil = new_cell("nil", x_env.symbol);

  x_env.frames = new_frame();

  bind("nil", x_env.nil, x_env.frames);
  bind("symbol", x_env.symbol, x_env.frames);
  bind("pair", x_env.pair, x_env.frames);

  x_env.binding = intern("binding");
  x_env.token = intern("token");
  x_env.builtin = intern("builtin");
  x_env.user = intern("user");
  x_env.true_ = intern("true");
  x_env.xector = intern("xector");
  x_env.int_ = intern("int");
  x_env.str = intern("str");

  x_env.fn0 = intern("fn0");
  x_env.fn1 = intern("fn1");
  x_env.fn2 = intern("fn2");
  x_env.fn3 = intern("fn3");

  x_env.dot = def_token(".");
  x_env.lparen = def_token("(");
  x_env.rparen = def_token(")");
  x_env.lbrack = def_token("[");
  x_env.rbrack = def_token("]");
  x_env.eof = def_token("EOF");

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
  cudaDeviceReset();
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
          if (expr == x_env.eof)
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
      if (expr == x_env.eof)
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
