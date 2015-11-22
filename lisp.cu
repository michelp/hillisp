#include "lisp.h"

void init(void) {
  x_dot = def_token(".");
  x_left = def_token("(");
  x_right = def_token(")");
  x_eof = def_token("EOF");

  x_nil = create_symbol("nil");
  for (int i = 0; i < HASH_TABLE_SIZE; i++)
    hash_table[i] = x_nil;
  enter(x_nil);
  x_true = intern("true");

  def_builtin("is", (void*)x_is, 2);
  def_builtin("car", (void*)x_car, 1);
  def_builtin("cdr", (void*)x_cdr, 1);
  def_builtin("cons", (void*)x_cons, 2);
  def_builtin("quote", (void*)x_quote, 1);
  def_builtin("cond", (void*)x_cond, 1);
  def_builtin("eval", (void*)x_eval, 1);
  def_builtin("apply", (void*)x_apply, 2);
  def_builtin("print", (void*)x_print, 1);
}

int main(int argc, const char* argv[]) {
  x_any expr;
  x_any value;

  init();
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
