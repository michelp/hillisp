#include "lisp.h"

__thread x_environ x_env;

x_any def_token(const char* new_name) {
  return new_cell(new_name, x_env.token);
}

x_any def_builtin(char const *name, void *fn, int num_args) {
  x_any cell;
  cell = intern(name);
  set_type(cell, x_env.builtin);
  set_val(cell, fn);
  switch(num_args) {
  case -1:
    set_type(cell, x_env.fnv);
    break;
  case 0:
    set_type(cell, x_env.fn0);
    break;
  case 1:
    set_type(cell, x_env.fn1);
    break;
  case 2:
    set_type(cell, x_env.fn2);
    break;
  case 3:
    set_type(cell, x_env.fn3);
    break;
  }
  return cell;
}

x_any def_special(char const *name, void *fn) {
  x_any cell;
  cell = intern(name);
  set_type(cell, x_env.special);
  set_val(cell, fn);
  return cell;
}

void init(void) {
  x_env.cell_pools = new_cell_pool(NULL);

  x_env.symbol = new_cell("symbol", NULL);
  set_type(x_env.symbol, x_env.symbol);

  x_env.pair = new_cell("pair", x_env.symbol);
  x_env.nil = new_cell("nil", x_env.symbol);
  x_env.binding = new_cell("binding", x_env.symbol);
  x_env.ns = new_cell("ns", x_env.symbol);

  init_frames();

  bind("symbol", x_env.symbol);
  bind("binding", x_env.binding);
  bind("ns", x_env.ns);
  bind("pair", x_env.pair);
  bind("nil", x_env.nil);

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
  x_env.fnv = intern("fnv");
  x_env.special = intern("special");

  x_env.dot = def_token(".");
  x_env.lparen = def_token("(");
  x_env.rparen = def_token(")");
  x_env.lbrack = def_token("[");
  x_env.rbrack = def_token("]");
  x_env.eof = def_token("EOF");

  def_builtin("is", (void*)x_is, 2);
  def_builtin("isinstance", (void*)x_isinstance, 2);
  def_builtin("type", (void*)x_type, 1);
  def_builtin("car", (void*)x_car, 1);
  def_builtin("cdr", (void*)x_cdr, 1);
  def_builtin("cons", (void*)x_cons, 2);
  def_builtin("list", (void*)x_list, -1);
  def_builtin("if", (void*)x_if, 1);
  def_builtin("while", (void*)x_while, 1);
  def_builtin("eval", (void*)x_eval, 1);
  def_builtin("apply", (void*)x_apply, 2);
  def_builtin("assert", (void*)x_assert, 1);
  def_builtin("print", (void*)x_print, 1);
  def_builtin("println", (void*)x_println, 1);
  def_builtin("+", (void*)x_add, 2);
  def_builtin("-", (void*)x_sub, 2);
  def_builtin("*", (void*)x_mul, 2);
  def_builtin("/", (void*)x_div, 2);
  def_builtin("fma", (void*)x_fma, 3);
  def_builtin("==", (void*)x_eq, 2);
  def_builtin("!=", (void*)x_neq, 2);
  def_builtin(">", (void*)x_gt, 2);
  def_builtin("<", (void*)x_lt, 2);
  def_builtin("not", (void*)x_not, 1);
  def_builtin("and", (void*)x_and, 2);
  def_builtin("all", (void*)x_all, 1);
  def_builtin("any", (void*)x_any_, 1);
  def_builtin("or", (void*)x_or, 2);
  def_builtin("fill", (void*)x_fill, 2);
  def_builtin("time", (void*)x_time, 0);
  def_builtin("gc", (void*)x_gc, 0);
  def_builtin("set", (void*)x_set, 2);
  def_builtin("dir", (void*)x_dir, 0);
  def_builtin("len", (void*)x_len, 1);

  def_special("quote", (void*)x_quote);
  def_special("def", (void*)x_def);
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
      printf("[%i]? ", x_env.frame_count);
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
  pop_frame();
  x_gc();
  x_env.result = cudaStreamDestroy(x_env.stream);
  CHECK;
  cudaDeviceReset();
  CHECK;
}
