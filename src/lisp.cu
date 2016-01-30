#include "lisp.h"

__thread x_environ x_env;

x_any def_token(const char* new_name) {
  return new_cell(new_name, x_env.token);
}

x_any def_fn(char const *name, void *fn) {
  x_any cell;
  cell = intern(name);
  set_type(cell, x_env.fn);
  set_val(cell, fn);
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

  x_env.int_ = intern("int");
  x_env.double_ = intern("double");
  x_env.dcomplex = intern("dcomplex");
  x_env.str = intern("str");

  x_env.ixector = intern("ixector");
  x_env.dxector = intern("dxector");
  x_env.dcxector = intern("dcxector");

  x_env.fn = intern("fn");
  x_env.special = intern("special");

  x_env.dot = def_token(".");
  x_env.dot = def_token("'");
  x_env.lparen = def_token("(");
  x_env.rparen = def_token(")");
  x_env.lbrack = def_token("[");
  x_env.rbrack = def_token("]");
  x_env.eof = def_token("EOF");

  def_fn("is", (void*)x_is);
  def_fn("isinstance", (void*)x_isinstance);
  def_fn("type", (void*)x_type);
  def_fn("car", (void*)x_car);
  def_fn("cdr", (void*)x_cdr);
  def_fn("cons", (void*)x_cons);
  def_fn("list", (void*)x_list);
  def_fn("eval", (void*)x_eval);
  def_fn("apply", (void*)x_apply);
  def_fn("assert", (void*)x_assert);
  def_fn("asserteq", (void*)x_asserteq);
  def_fn("assertall", (void*)x_assertall);
  def_fn("assertany", (void*)x_assertany);
  def_fn("print", (void*)x_print);
  def_fn("println", (void*)x_println);
  def_fn("printsp", (void*)x_printsp);
  def_fn("+", (void*)x_add);
  def_fn("+=", (void*)x_addass);
  def_fn("-", (void*)x_sub);
  def_fn("-=", (void*)x_subass);
  def_fn("*", (void*)x_mul);
  def_fn("*=", (void*)x_mulass);
  def_fn("/", (void*)x_div);
  def_fn("/=", (void*)x_divass);
  def_fn("fma", (void*)x_fma);
  def_fn("fma=", (void*)x_fmaass);
  def_fn("==", (void*)x_eq);
  def_fn("!=", (void*)x_neq);
  def_fn(">", (void*)x_gt);
  def_fn("<", (void*)x_lt);
  def_fn(">=", (void*)x_gte);
  def_fn("<=", (void*)x_lte);
  def_fn("complex", (void*)x_complex);
  def_fn("not", (void*)x_not);
  def_fn("and", (void*)x_and);
  def_fn("all", (void*)x_all);
  def_fn("any", (void*)x_any_);
  def_fn("or", (void*)x_or);
  def_fn("fill", (void*)x_fill);
  def_fn("empty", (void*)x_empty);
  def_fn("time", (void*)x_time);
  def_fn("gc", (void*)x_gc);
  def_fn("dir", (void*)x_dir);
  def_fn("len", (void*)x_len);
  def_fn("range", (void*)x_range);
  def_fn("collect", (void*)x_collect);

  def_special("quote", (void*)x_quote);
  def_special("def", (void*)x_def);
  def_special("if", (void*)x_if);
  def_special("while", (void*)x_while);
  def_special("do", (void*)x_do);
  def_special("for", (void*)x_for);
  def_special("set", (void*)x_set);
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
          value = eval(expr);
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
      value = eval(expr);
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
